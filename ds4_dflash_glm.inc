/* GLM-5.3 DFlash speculative cycle: draft (block-diffusion drafter) ->
 * one batched target verify with per-step KDA snapshots and tap-layer
 * feature capture -> longest-prefix accept. Greedy only (v1). Lossless:
 * every committed draft equals the target argmax on the true prefix, and
 * s->logits is left at the new frontier (the bonus distribution), mirroring
 * the MTP spec-cycle contract. */

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
static bool glm53_graph_restore_kda_stepsnap(
        ds4_glm_gpu_graph *g,
        ds4_gpu_tensor *snap,
        uint32_t step,
        uint64_t step_bytes) {
    if (!g || !g->glm53 || !snap) return false;
    const uint64_t expected = glm53_graph_kda_state_bytes(g);
    if (expected == 0 || step_bytes < expected) return false;
    bool ok = glm_graph_begin_commands_if_needed();
    const uint64_t offset = (uint64_t)step * step_bytes;
    uint64_t walked = 0;
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        if (!ds4_glm53_layer_is_kda(il)) continue;
        ds4_gpu_tensor *state[2] = {
            g->layer_kda_conv_state[il],
            g->layer_kda_recurrent_state[il],
        };
        for (uint32_t i = 0; ok && i < 2; i++) {
            const uint64_t bytes = ds4_gpu_tensor_bytes(state[i]);
            ok = ds4_gpu_tensor_copy(state[i], 0, snap, offset + walked,
                                     bytes) != 0;
            walked += bytes;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    return ok && walked == expected;
}

/* temperature <= 0 or rng == NULL: greedy longest-prefix accept (byte-exact
 * vs serial argmax). temperature > 0: lossless rejection sampling — the
 * drafter is deterministic (point-mass q), so each draft d is accepted with
 * probability p(d) under the temp/top-k/top-p-filtered target distribution;
 * on rejection the frontier logits get d masked to -inf so the caller's own
 * sampling draws from the exact residual distribution. */
static int ds4_session_glm_dflash_cycle_impl(
        ds4_session *s, int first_token,
        float temperature, int top_k, float top_p, float min_p,
        uint64_t *rng,
        int *accepted, int accepted_cap,
        char *err, size_t errlen) {
    ds4_engine *e = s->engine;
    ds4_glm_gpu_graph *g = &s->glm_graph;
    const ds4_dflash2_weights *dw = &e->dflash_weights;
    const uint32_t pos = (uint32_t)s->checkpoint.len;
    const uint32_t H = DS4_N_EMBD;
    const uint32_t n_taps = dw->n_target;

    /* UNSUPPORTED ON THIS BRANCH: incomplete speculative rollback.
     *
     * The cycle below snapshots and restores the KDA conv/recurrent state
     * only (glm53_graph_restore_kda_stepsnap, sized by
     * glm53_graph_kda_state_bytes).  The speculative state of a GLM-5.3 graph
     * is wider than that: every verify route this cycle can take --
     * glm53_graph_verify_rows, glm_graph_forward_tokens and
     * glm_graph_forward_indexed_tokens -- also drives
     * ds4_gpu_glm53_indexer_pool_update_tensor over
     * layer_indexer_tail_k[il] / layer_indexer_tail_gate[il] for all L
     * drafted rows, and glm53_graph_spec_state_tensors counts that tail
     * (the K object owns the contiguous K+gate allocation; the gate is a
     * view into it) as part of the state a rejected draft must undo.  After a
     * partial acceptance the tail therefore still carries the rejected rows,
     * while the checkpoint is marked valid and the frontier advances.
     *
     * Rather than leave a selectable mode whose rollback is known to be
     * incomplete, refuse the whole cycle -- before any target state is
     * mutated -- whenever the graph has indexer tail state, and let the
     * caller's ordinary serial path evaluate the token.  The predicate is the
     * same one that guards the MTP row-snapshot fast path: the snapshot is
     * usable only when it covers the entire speculative state.
     *
     * To lift this: capture and restore the indexer tail alongside the KDA
     * state per draft step (a Metal-side change in
     * ds4_gpu_glm53_kda_stepsnap_begin and its consumers), or restore the
     * full speculative state and replay the committed prefix before
     * returning.  Either fix must be certified by a forced-rejection run
     * across a pool/tail boundary against a serial continuation. */
    {
        static int refuse_reported;
        if (glm53_graph_spec_state_bytes(g) != glm53_graph_kda_state_bytes(g)) {
            if (!refuse_reported) {
                refuse_reported = 1;
                fprintf(stderr,
                        "ds4: dflash disabled on this graph: speculative "
                        "rollback does not restore the DSA indexer tail "
                        "(unsupported on this branch); decoding serially\n");
            }
            const int rc = ds4_session_eval_internal(s, first_token, false,
                                                     err, errlen);
            if (rc != 0) return -1;
            accepted[0] = first_token;
            return 1;
        }
    }

    s->glm_mtp_rollback_valid = false;
    s->glm_mtp_have = 0;
    if (s->dflash_last_pos != 0 && pos < s->dflash_last_pos) {
        /* The session rewound (server slot reuse across requests, or a
         * thinking-trim checkpoint rewind): the drafter's context rows are
         * positioned for the old generation and the throttle state — EMA,
         * probe backoff — belongs to the old content. Start fresh. */
        s->dflash_ema = 0.0f;
        s->dflash_lowrun = 0;
        s->dflash_probe_fails = 0;
        s->dflash_cycles = 0;
        s->dflash_pending_rows = 0;
        s->dflash_warm = 0;
        /* keep dflash_cycle_ms/serial_ms: cost tracking survives requests
         * (position barely moves across a rewound slot) */
        for (uint32_t i = 0; i < s->dflash_cache.n_layer; i++) {
            s->dflash_cache.layer[i].len = 0;
        }
    }
    s->dflash_last_pos = pos;

    uint32_t L_max = dw->block_size;
    if (L_max > DS4_DFLASH2_MAX_DRAFT + 1u) L_max = DS4_DFLASH2_MAX_DRAFT + 1u;
    if ((int)L_max > accepted_cap) L_max = (uint32_t)accepted_cap;
    if (pos + L_max > g->ctx_size) {
        L_max = g->ctx_size > pos ? g->ctx_size - pos : 0u;
    }
    if (g->compact_cache_cap != 0 && pos + L_max >= g->compact_cache_cap) {
        L_max = g->compact_cache_cap > pos + 1u ?
            g->compact_cache_cap - pos - 1u : 0u;
    }
    if (L_max == 0) {
        const int rc = ds4_session_eval_internal(s, first_token, false,
                                                 err, errlen);
        if (rc != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }

    /* Lazy scratch: per-step KDA snapshots (8 x state), feature capture
     * (8 x taps x embd), the drafter's sliding context-KV cache, and the
     * pending-feature staging rows. */
    const uint64_t state_bytes = glm53_graph_kda_state_bytes(g);
    if (!g->dflash_stepsnap && state_bytes != 0) {
        g->dflash_stepsnap = ds4_gpu_tensor_alloc(
                (uint64_t)(DS4_DFLASH2_MAX_DRAFT + 1u) * state_bytes);
    }
    if (!g->dflash_capture) {
        g->dflash_capture = ds4_gpu_tensor_alloc(
                (uint64_t)(DS4_DFLASH2_MAX_DRAFT + 1u) * n_taps * H *
                sizeof(float));
    }
    if (!g->dflash_stepsnap || !g->dflash_capture) {
        snprintf(err, errlen, "dflash: scratch alloc failed");
        return -1;
    }
    if (s->dflash_cache.n_layer == 0) {
        /* the drafter's attention cost grows ~linearly with context rows
         * (~0.12ms/row); past ~256 rows the extra acceptance no longer pays
         * for the added draft latency. */
        uint32_t ccap = dw->sliding_window ? dw->sliding_window - 1u : 2047u;
        if (ccap > 256u) ccap = 256u;
        const char *cenv = getenv("DS4_DFLASH_CTX_CAP");
        if (cenv && atoi(cenv) > 0) ccap = (uint32_t)atoi(cenv);
        if (!dflash_cache_init(&s->dflash_cache, dw, ccap)) {
            snprintf(err, errlen, "dflash: cache alloc failed");
            return -1;
        }
    }
    if (!s->dflash_pending) {
        s->dflash_pending =
            xmalloc((size_t)(DS4_DFLASH2_MAX_DRAFT + 1u) * n_taps * H *
                    sizeof(float));
        s->dflash_pending_rows = 0;
    }

    /* First cycle: ingest the prefill feature ring (last <=2047 prompt
     * positions) into the drafter's context cache so drafting starts with
     * the prompt in view. The throwaway draft block costs one drafter pass. */
    const double t_cycle0 = getenv("DS4_DFLASH_STATS") ? now_sec() : 0.0;
    int drafts[DS4_DFLASH2_MAX_DRAFT];
    int n_draft = 0;
    if (!s->dflash_warm) {
        if (g_glm_dflash_seed.ring_len != 0 &&
            g_glm_dflash_seed.ring_end_pos == pos) {
            int seed_drafts[DS4_DFLASH2_MAX_DRAFT];
            (void)dflash2_propose(seed_drafts, 1,
                                  &e->dflash_model, dw,
                                  &e->model, &e->weights,
                                  g_glm_dflash_seed.ring,
                                  g_glm_dflash_seed.ring_len,
                                  &s->dflash_cache,
                                  first_token, pos);
            g_glm_dflash_seed.ring_len = 0;
        }
        s->dflash_warm = 1;
        /* consumed (or unusable) — an empty ring is never re-ingested */
        g_glm_dflash_seed.ring_len = 0;
    }
    /* Adaptive throttle with live-measured break-even: cycle and serial
     * token costs are tracked per session (EMA of wall time), so the park
     * threshold prices depth (verify and serial both slow with position),
     * content, and kernel changes automatically instead of relying on
     * hand-tuned constants. Below threshold, run plain decode and probe
     * with a full block on an exponentially backed-off cadence. */
    const float breakeven =
        s->dflash_serial_ms > 0.1f && s->dflash_cycle_ms > 0.1f ?
            s->dflash_cycle_ms / s->dflash_serial_ms : 4.1f;
    if (!getenv("DS4_DFLASH_NO_ADAPTIVE") && s->dflash_warm) {
        if (s->dflash_ema == 0.0f) s->dflash_ema = breakeven + 1.5f;
        if (s->dflash_ema < breakeven) {
            s->dflash_lowrun++;
            uint32_t window = 64u << (s->dflash_probe_fails < 4u ?
                                      s->dflash_probe_fails : 4u);
            /* probes come in pairs: the first refreshes drafter
             * features/context after a serial stretch, the second measures
             * honestly. lowrun==1 also probes immediately after the EMA
             * first dips, while features are still fresh. */
            if (s->dflash_lowrun % window >= 2u) {
                const double fb_t0 = now_sec();
                const int rc = ds4_session_eval_internal(s, first_token,
                                                         false, err, errlen);
                if (rc != 0) return -1;
                const float fb_ms = (float)((now_sec() - fb_t0) * 1000.0);
                s->dflash_serial_ms = s->dflash_serial_ms <= 0.0f ? fb_ms :
                    0.9f * s->dflash_serial_ms + 0.1f * fb_ms;
                if (getenv("DS4_DFLASH_STATS")) {
                    static uint32_t fbn;
                    if (++fbn % 128u == 0u) {
                        fprintf(stderr,
                                "ds4: dflash fallback ema=%.2fms n=%u\n",
                                s->dflash_serial_ms, fbn);
                    }
                }
                s->dflash_pending_rows = 0;
                accepted[0] = first_token;
                return 1;
            }
            /* probe window: fall through into a full cycle */
        } else {
            s->dflash_lowrun = 0;
        }
    }
    const double cyc_t0 = now_sec();
    uint32_t l_clamp = L_max;
    {
        const char *fd = getenv("DS4_DFLASH_FORCE_DRAFTS");
        if (fd && fd[0]) {
            int v = atoi(fd);
            if (v >= 0 && (uint32_t)(v + 1) < l_clamp) l_clamp = (uint32_t)(v + 1);
        }
    }
    if (getenv("DS4_DFLASH_ZERO_FEATURES") && s->dflash_pending_rows) {
        memset(s->dflash_pending, 0,
               (size_t)s->dflash_pending_rows * n_taps * H * sizeof(float));
    }
    if (s->dflash_warm && s->dflash_pending_rows != 0 && l_clamp > 1u) {
        n_draft = dflash2_propose(drafts, (int)(l_clamp - 1u),
                                  &e->dflash_model, dw,
                                  &e->model, &e->weights,
                                  s->dflash_pending,
                                  s->dflash_pending_rows,
                                  &s->dflash_cache,
                                  first_token, pos);
        if (n_draft < 0) n_draft = 0;
        s->dflash_pending_rows = 0;
    }
    if (n_draft == 0) {
        /* Nothing drafted (feature-refresh probe, or the drafter proposed
         * nothing): run the exact serial decode with tap capture armed so
         * the next cycle gets fresh features without paying the batch
         * verify's fixed cost (~70ms vs ~34ms serial). */
        g_glm_dflash_cap.enabled = 1;
        g_glm_dflash_cap.buf = g->dflash_capture;
        g_glm_dflash_cap.n_rows = 1;
        g_glm_dflash_cap.n_taps = n_taps;
        for (uint32_t j = 0; j < n_taps; j++) {
            g_glm_dflash_cap.taps[j] = dw->target_layers[j];
        }
        const int rc = ds4_session_eval_internal(s, first_token, false,
                                                 err, errlen);
        g_glm_dflash_cap.enabled = 0;
        if (rc != 0) return -1;
        if (ds4_gpu_tensor_read(g->dflash_capture, 0, s->dflash_pending,
                                (uint64_t)n_taps * H * sizeof(float))) {
            s->dflash_pending_rows = 1;
        } else {
            s->dflash_pending_rows = 0;
        }
        accepted[0] = first_token;
        return 1;
    }
    const uint32_t L = (uint32_t)n_draft + 1u;
    int toks[DS4_DFLASH2_MAX_DRAFT + 1];
    toks[0] = first_token;
    for (int i = 0; i < n_draft; i++) toks[i + 1] = drafts[i];

    g_glm_dflash_cap.enabled = 1;
    g_glm_dflash_cap.buf = g->dflash_capture;
    g_glm_dflash_cap.n_rows = L;
    g_glm_dflash_cap.n_taps = n_taps;
    for (uint32_t j = 0; j < n_taps; j++) {
        g_glm_dflash_cap.taps[j] = dw->target_layers[j];
    }
    const double t_draft = getenv("DS4_DFLASH_STATS") ? now_sec() : 0.0;
    ds4_gpu_glm53_kda_stepsnap_begin(g->dflash_stepsnap,
                                     (uint32_t)(state_bytes / sizeof(float)),
                                     DS4_DFLASH2_MAX_DRAFT + 1u);
    static float *hc_rows_host = NULL;
    if (!hc_rows_host) {
        hc_rows_host = xmalloc((size_t)(DS4_DFLASH2_MAX_DRAFT + 1u) *
                               DS4_N_HC * H * sizeof(float));
    }
    bool verified;
    /* Experimental, default OFF: measured byte-identical to the batch
     * path but NOT faster (~105 vs ~107ms) — once the batch path gained
     * flush-every-8-layers encode overlap, the remaining verify cost is
     * genuine GPU kernel time, which this pass shares. Kept as the
     * skeleton for a future fused-rows-kernel push (hc_pre norm+mix rows
     * fusion, KDA prefill occupancy); also limited to pos+n <= 4096. */
    if (getenv("DS4_DFLASH_VERIFY_ROWS") &&
        glm53_graph_verify_rows_ready(g, pos, L)) {
        /* Decode-style verify: one command buffer, causal DSA over the
         * compact caches. The ready check precedes any GPU work, so a
         * false return here is a hard error, not a fallback — KDA state
         * may already have advanced, and re-running the batch path would
         * advance it a second time. */
        verified = glm53_graph_verify_rows(g, &e->model, &e->weights,
                                           toks, pos, L, hc_rows_host, NULL);
    } else if (!glm53_graph_use_indexed_prefill(g) &&
               glm_graph_span_fits_full_attention(g, pos, L)) {
        verified = glm_graph_forward_tokens(g, &e->model, &e->weights,
                                            toks, NULL, NULL, 0,
                                            pos, L, hc_rows_host, NULL,
                                            NULL, NULL, pos, 0, L);
    } else {
        verified = glm_graph_forward_indexed_tokens(g, &e->model, &e->weights,
                                                    toks, NULL, NULL, 0,
                                                    pos, L, hc_rows_host, NULL,
                                                    NULL, NULL, pos, 0, L);
    }
    ds4_gpu_glm53_kda_rowsnap_end();
    g_glm_dflash_cap.enabled = 0;
    const double t_verify = getenv("DS4_DFLASH_STATS") ? now_sec() : 0.0;
    if (!verified) {
        snprintf(err, errlen, "dflash: verify forward failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Per-row head through the exact single-row path (bit-identical to
     * serial decode's head), logits read to the host per row. */
    static float *row_logits = NULL;
    if (!row_logits) {
        row_logits = xmalloc((size_t)(DS4_DFLASH2_MAX_DRAFT + 1u) *
                             DS4_N_VOCAB * sizeof(float));
    }
    const uint64_t hc_row = (uint64_t)DS4_N_HC * H;
    if (!g->dflash_hcrows) {
        g->dflash_hcrows = ds4_gpu_tensor_alloc(
                (uint64_t)(DS4_DFLASH2_MAX_DRAFT + 1u) * hc_row *
                sizeof(float));
    }
    if (!g->dflash_hcrows ||
        !ds4_gpu_tensor_write(g->dflash_hcrows, 0, hc_rows_host,
                              (uint64_t)L * hc_row * sizeof(float))) {
        snprintf(err, errlen, "dflash: hc upload failed");
        s->checkpoint_valid = false;
        return -1;
    }
    /* All L heads in one command stream: collapse each hc row, then the
     * shared Q8 head into the pool logits, one readback at the end. */
    {
        bool ok = g_dflash_pool.inited &&
                  L <= g_dflash_pool.spec_cap &&
                  ds4_gpu_begin_commands() != 0;
        for (uint32_t r = 0; ok && r < L; r++) {
            ds4_gpu_tensor *row = glm_graph_tensor_row_view_strided(
                    g->dflash_hcrows, r, hc_row, hc_row);
            ds4_gpu_tensor *dst = ds4_gpu_tensor_view(
                    g_dflash_pool.batch_norm,
                    (uint64_t)r * H * sizeof(float),
                    (uint64_t)H * sizeof(float));
            ok = row && dst &&
                 ds4_gpu_hc_weighted_sum_tensor(dst, row, g->hc_mean_weights,
                                                DS4_N_EMBD, DS4_N_HC) != 0;
            ds4_gpu_tensor_free(row);
            ds4_gpu_tensor_free(dst);
        }
        if (ok) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(
                    g_dflash_pool.batch_norm, g_dflash_pool.batch_norm,
                    e->model.map, e->model.size,
                    e->weights.output_norm->abs_offset,
                    DS4_N_EMBD, L, DS4_RMS_EPS) != 0;
        }
        if (ok) {
            for (uint32_t r = 0; ok && r < L; r++) {
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(
                        g_dflash_pool.batch_norm,
                        (uint64_t)r * H * sizeof(float),
                        (uint64_t)H * sizeof(float));
                ds4_gpu_tensor *yr = ds4_gpu_tensor_view(
                        g_dflash_pool.batch_logits,
                        (uint64_t)r * DS4_N_VOCAB * sizeof(float),
                        (uint64_t)DS4_N_VOCAB * sizeof(float));
                ok = xr && yr &&
                     ds4_gpu_matmul_quant_tensor(
                             yr, e->model.map, e->model.size,
                             e->weights.output->abs_offset,
                             e->weights.output->type,
                             H, DS4_N_VOCAB, xr, 1);
                ds4_gpu_tensor_free(xr);
                ds4_gpu_tensor_free(yr);
            }
        }
        if (!ds4_gpu_end_commands()) ok = false;
        if (ok) {
            ok = ds4_gpu_tensor_read(
                    g_dflash_pool.batch_logits, 0, row_logits,
                    (uint64_t)L * DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (!ok) {
            snprintf(err, errlen, "dflash: verify head failed");
            s->checkpoint_valid = false;
            return -1;
        }
    }

    const double t_heads = getenv("DS4_DFLASH_STATS") ? now_sec() : 0.0;
    /* Accept the longest matching prefix; frontier = row[n_accept]. */
    int n_accept = 0;
    int rejected_tok = -1;
    if (temperature > 0.0f && rng && s->sample_probs) {
        for (int i = 0; i < n_draft; i++) {
            if (!sample_build_probabilities(
                    row_logits + (size_t)i * DS4_N_VOCAB,
                    DS4_N_VOCAB, temperature, top_k, top_p, min_p,
                    s->sample_probs)) {
                snprintf(err, errlen, "dflash: target distribution failed");
                s->checkpoint_valid = false;
                return -1;
            }
            if (!speculative_point_accept(s->sample_probs[drafts[i]], 1.0f,
                                          rng)) {
                rejected_tok = drafts[i];
                break;
            }
            n_accept++;
        }
    } else {
        for (int i = 0; i < n_draft; i++) {
            const int pred = glm_session_logits_argmax(
                    row_logits + (size_t)i * DS4_N_VOCAB);
            if (pred != drafts[i]) break;
            n_accept++;
        }
    }
    const uint32_t n_committed = (uint32_t)n_accept + 1u;
    memcpy(s->logits, row_logits + (size_t)n_accept * DS4_N_VOCAB,
           (size_t)DS4_N_VOCAB * sizeof(float));
    if (rejected_tok >= 0) s->logits[rejected_tok] = -1e30f;

    if (n_committed < L) {
        if (!glm53_graph_restore_kda_stepsnap(g, g->dflash_stepsnap,
                                              n_committed - 1u,
                                              state_bytes)) {
            snprintf(err, errlen, "dflash: KDA step restore failed");
            s->checkpoint_valid = false;
            return -1;
        }
    }
    for (uint32_t i = 0; i < n_committed; i++) {
        token_vec_push(&s->checkpoint, toks[i]);
        accepted[i] = toks[i];
    }
    s->checkpoint_valid = true;
    ds4_session_glm_note_dense_cache(s, pos, n_committed);

    /* Stage committed rows' features for the next cycle's draft context. */
    if (ds4_gpu_tensor_read(g->dflash_capture, 0, s->dflash_pending,
                            (uint64_t)n_committed * n_taps * H *
                                sizeof(float))) {
        s->dflash_pending_rows = n_committed;
    } else {
        s->dflash_pending_rows = 0;
    }
    s->dflash_warm = 1;
    if (n_draft > 0) {
        const float alpha = 0.15f;
        if (s->dflash_ema == 0.0f) s->dflash_ema = 5.5f;
        if (s->dflash_cycles++ == 0u && n_committed <= 3u) {
            /* the very first full cycle already shows weak acceptance:
             * park immediately instead of paying 3-4 more losing cycles
             * while the EMA descends from its optimistic init. */
            s->dflash_ema = (float)n_committed;
        }
        s->dflash_ema = (1.0f - alpha) * s->dflash_ema +
                        alpha * (float)n_committed;
        if (s->dflash_lowrun != 0 && s->dflash_lowrun % 64u != 0u) {
            /* fresh (second-of-pair or immediate) probe: read the regime
             * from the probe itself — anything that beats the measured
             * break-even re-engages immediately */
            if ((float)n_committed >= breakeven + 0.5f) {
                s->dflash_ema = (float)n_committed;
                s->dflash_probe_fails = 0;
            } else {
                s->dflash_probe_fails++;
            }
        }
        if (s->dflash_ema >= breakeven) {
            s->dflash_lowrun = 0;
            s->dflash_probe_fails = 0;
        }
    }
    if (getenv("DS4_DFLASH_STATS")) {
        fprintf(stderr,
                "ds4: dflash cycle drafted=%d accepted=%d ctx=%u "
                "draft=%.1fms verify=%.1fms heads=%.1fms tail=%.1fms "
                "serial=%.1fms cyc=%.1fms be=%.2f\n",
                n_draft, n_accept,
                s->dflash_cache.n_layer ? s->dflash_cache.layer[0].len : 0u,
                (t_draft - t_cycle0) * 1000.0,
                (t_verify - t_draft) * 1000.0,
                (t_heads - t_verify) * 1000.0,
                (now_sec() - t_heads) * 1000.0,
                s->dflash_serial_ms, s->dflash_cycle_ms, breakeven);
    }
    {
        const float cyc_ms = (float)((now_sec() - cyc_t0) * 1000.0);
        s->dflash_cycle_ms = s->dflash_cycle_ms <= 0.0f ? cyc_ms :
            0.9f * s->dflash_cycle_ms + 0.1f * cyc_ms;
    }
    return (int)n_committed;
}

static int ds4_session_glm_dflash_cycle_argmax(
        ds4_session *s, int first_token,
        int *accepted, int accepted_cap,
        char *err, size_t errlen) {
    return ds4_session_glm_dflash_cycle_impl(s, first_token,
                                             0.0f, 0, 0.0f, 0.0f, NULL,
                                             accepted, accepted_cap,
                                             err, errlen);
}
#endif
