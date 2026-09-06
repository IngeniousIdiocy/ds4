/* GLM-5.3 DFlash speculative cycle: draft (block-diffusion drafter) ->
 * one batched target verify with per-step state snapshots and tap-layer
 * feature capture -> longest-prefix accept. Every committed draft is
 * verified against the target distribution on its causal prefix, and
 * s->logits is left at the new frontier (the bonus distribution), mirroring
 * the MTP spec-cycle contract. */

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
#include "ds4_dflash_script.inc"

static bool glm53_graph_restore_dflash_stepsnap(
        ds4_glm_gpu_graph *g,
        ds4_gpu_tensor *snap,
        uint32_t step,
        uint64_t step_bytes) {
    if (!g || !g->glm53 || !snap) return false;
    const uint64_t expected = glm53_graph_spec_state_bytes(g);
    if (expected == 0 || step_bytes != expected ||
        step >= ds4_gpu_tensor_bytes(snap) / step_bytes) return false;
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
    /* Capture layout puts tails after all KDA layers, each K allocation
     * includes its gate view. It is deliberately distinct from the
     * interleaved pre-block backup layout. */
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        if (ds4_glm53_layer_is_kda(il)) continue;
        ds4_gpu_tensor *tail = g->layer_indexer_tail_k[il];
        const uint64_t bytes = tail ? ds4_gpu_tensor_bytes(tail) : 0;
        if (bytes) ok = ds4_gpu_tensor_copy(tail, 0, snap, offset + walked,
                                           bytes) != 0;
        walked += bytes;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    return ok && walked == expected;
}

/* temperature <= 0 or rng == NULL: greedy longest-prefix accept against
 * batched target logits. Batched arithmetic can differ from serial.
 * temperature > 0: lossless rejection sampling against that distribution — the
 * drafter is deterministic (point-mass q), so each draft d is accepted with
 * probability p(d) under the temp/top-k/top-p-filtered target distribution;
 * on rejection the frontier logits get d masked to -inf so the caller's own
 * sampling draws from the exact residual distribution. */
/* Failure exit after the verify forward has advanced per-token state: put the
 * complete speculative state back if we saved it, and keep the checkpoint only
 * when that restore actually happened (dflash_exit_keeps_checkpoint). */
static void dflash_cycle_fail_after_verify(ds4_session *s,
                                           ds4_glm_gpu_graph *g,
                                           bool state_saved) {
    bool restore_ok = false;
    if (state_saved) {
        restore_ok = glm53_graph_copy_spec_state_to(g, g->dflash_state_backup,
                                                    false);
    }
    s->checkpoint_valid =
        dflash_exit_keeps_checkpoint(true, state_saved, restore_ok);
}

/* Undo the process-global instrumentation this cycle arms: the tap capture
 * and the per-step KDA snapshot. Safe to call when neither is armed. */
static void dflash_cycle_disarm(void) {
    g_glm_dflash_cap.enabled = 0;
    g_glm_dflash_cap.buf = NULL;
    g_glm_dflash_cap.n_rows = 0;
    ds4_gpu_glm53_kda_rowsnap_end();
    ds4_gpu_glm53_tail_stepsnap_end();
}

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

    /* Each verify captures KDA post-row state and DSA tails reconstructed
     * from the pre-block tail plus only that prefix's raw K/gates. Restore
     * committed-1 on partial acceptance. The complete pre-block backup is
     * retained for failure recovery or missing capture coverage.
     * DS4_DFLASH_FORCE_REPLAY forces restore+serial replay on EVERY cycle,
     * including full acceptance: an oracle for state on accepted token IDs,
     * not a claim that batched argmax always equals serial argmax.
     * DS4_DFLASH_NO_WIDE_ROLLBACK still declines wide graphs before verify. */
    const uint64_t spec_bytes = glm53_graph_spec_state_bytes(g);
    const uint64_t kda_bytes = glm53_graph_kda_state_bytes(g);
    const bool wide_state = dflash_state_is_wide(spec_bytes, kda_bytes);
    const bool allow_wide = getenv("DS4_DFLASH_NO_WIDE_ROLLBACK") == NULL;
    const bool force_replay = getenv("DS4_DFLASH_FORCE_REPLAY") != NULL;
    if (wide_state && !allow_wide) {
        static int refuse_reported;
        if (!refuse_reported) {
            refuse_reported = 1;
            fprintf(stderr,
                    "ds4: dflash disabled on this graph: "
                    "DS4_DFLASH_NO_WIDE_ROLLBACK is set and the speculative "
                    "state includes the DSA indexer tail; decoding "
                    "serially\n");
        }
        const int rc = ds4_session_eval_internal(s, first_token, false,
                                                 err, errlen);
        if (rc != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }

    s->glm_mtp_rollback_valid = false;
    s->glm_mtp_have = 0;
    if (dflash_conditioning_stale(s->dflash_bound_gen, s->dflash_gen,
                                  pos, s->dflash_last_pos)) {
        /* Either the prefix is no longer an extension of the one the drafter
         * was warmed on (a different request on a reused slot, a restored
         * payload, an invalidation), or the frontier moved backwards inside
         * this generation (a thinking-trim rewind). The drafter's context
         * rows and the throttle state — EMA, probe backoff — belong to the
         * old content either way. Start fresh.
         *
         * Costs are reseeded as well: a restored/new request can be at a
         * completely different depth. */
        ds4_session_dflash_reset_conditioning(s);
        s->dflash_bound_gen = s->dflash_gen;
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

    /* Lazy scratch: complete per-step snapshots (8 x state), feature capture
     * (8 x taps x embd), the drafter's sliding context-KV cache, and the
     * pending-feature staging rows. */
    const uint64_t state_bytes = spec_bytes;
    if (!g->dflash_stepsnap && state_bytes != 0) {
        g->dflash_stepsnap = ds4_gpu_tensor_alloc(
                (uint64_t)(DS4_DFLASH2_MAX_DRAFT + 1u) * state_bytes);
    }
    const bool want_backup = wide_state || force_replay;
    if (want_backup && !g->dflash_state_backup && spec_bytes != 0) {
        g->dflash_state_backup = ds4_gpu_tensor_alloc(spec_bytes);
    }
    if (!g->dflash_capture) {
        g->dflash_capture = ds4_gpu_tensor_alloc(
                (uint64_t)(DS4_DFLASH2_MAX_DRAFT + 1u) * n_taps * H *
                sizeof(float));
    }
    if ((state_bytes != 0 && !g->dflash_stepsnap) || !g->dflash_capture ||
        (want_backup && !g->dflash_state_backup)) {
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

    /* Deterministic draft supply (test-only, see ds4_dflash_script.inc). A
     * script names the positions that get a speculative block and how many of
     * its drafts are the true continuation; every other position -- and every
     * position in the control arm -- decodes serially, so the frontier lands
     * exactly where the next block asked for it. */
    dflash_script_init();
    const ds4_dflash_script_block *sblk = NULL;
    if (g_dflash_script.active) {
        sblk = g_dflash_script.serial ? NULL : dflash_script_at(pos);
        if (sblk && (uint32_t)sblk->n + 1u > L_max) sblk = NULL;
        if (!sblk) {
            g_glm_dflash_cap.enabled = 1;
            g_glm_dflash_cap.buf = g->dflash_capture;
            g_glm_dflash_cap.n_rows = 1;
            g_glm_dflash_cap.n_taps = n_taps;
            for (uint32_t j = 0; j < n_taps; j++) {
                g_glm_dflash_cap.taps[j] = dw->target_layers[j];
            }
            const int rc = ds4_session_eval_internal(s, first_token, false,
                                                     err, errlen);
            dflash_cycle_disarm();
            if (rc != 0) return -1;
            if (ds4_gpu_tensor_read(g->dflash_capture, 0, s->dflash_pending,
                                    (uint64_t)n_taps * H * sizeof(float))) {
                s->dflash_pending_rows = 1;
            } else {
                s->dflash_pending_rows = 0;
            }
            s->dflash_warm = 1;
            dflash_script_record_id(pos, first_token);
            if (dflash_script_dump_frontier(pos + 1u)) {
                dflash_script_dump_logits(pos + 1u, s->logits);
                dflash_script_dump_state(pos + 1u, g, "serial");
            }
            accepted[0] = first_token;
            return 1;
        }
    }

    /* First cycle: ingest the prefill feature ring (last <=2047 prompt
     * positions) into the drafter's context cache so drafting starts with
     * the prompt in view. The throwaway draft block costs one drafter pass. */
    const double t_cycle0 = getenv("DS4_DFLASH_STATS") ? now_sec() : 0.0;
    int drafts[DS4_DFLASH2_MAX_DRAFT];
    int n_draft = 0;
    if (!s->dflash_warm) {
        const double ingest_t0 = now_sec();
        if (dflash_seed_ring_usable(g_glm_dflash_seed.owner_gen,
                                    s->dflash_gen,
                                    g_glm_dflash_seed.ring_len,
                                    g_glm_dflash_seed.ring_end_pos, pos)) {
            int seed_drafts[DS4_DFLASH2_MAX_DRAFT];
            (void)dflash2_propose(seed_drafts, 1,
                                  &e->dflash_model, dw,
                                  &e->model, &e->weights,
                                  g_glm_dflash_seed.ring,
                                  g_glm_dflash_seed.ring_len,
                                  &s->dflash_cache,
                                  first_token, pos);
        }
        s->dflash_warm = 1;
        /* consumed (or unusable) — a ring is never ingested twice, and one
         * left by another session's prefill is dropped rather than reused */
        glm_dflash_seed_invalidate();
        /* paid before cyc_t0 below, so the throttle never sees it */
        g_dflash_overhead.ingest_ms += (now_sec() - ingest_t0) * 1000.0;
        g_dflash_overhead.ingests++;
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
    if (!sblk && !getenv("DS4_DFLASH_NO_ADAPTIVE") && s->dflash_warm) {
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
                s->dflash_serial_ms = dflash_timing_ema(s->dflash_serial_ms,
                                                       fb_ms);
                g_dflash_overhead.park_ms += (double)fb_ms;
                g_dflash_overhead.parks++;
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
    if (sblk) {
        /* The verifier is untouched: these are proposals, and the target's
         * own argmax decides which of them survive. */
        n_draft = dflash_script_drafts(sblk, pos, drafts, DS4_N_VOCAB);
        s->dflash_pending_rows = 0;
    } else if (s->dflash_warm && s->dflash_pending_rows != 0 && l_clamp > 1u) {
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
        const double refresh_t0 = now_sec();
        const int rc = ds4_session_eval_internal(s, first_token, false,
                                                 err, errlen);
        dflash_cycle_disarm();
        const float refresh_ms = (float)((now_sec() - refresh_t0) * 1000.0);
        g_dflash_overhead.refresh_ms += refresh_ms;
        g_dflash_overhead.refreshes++;
        if (rc != 0) return -1;
        /* This is a measured serial target step even when no park has yet
         * occurred. Capture overhead stays charged conservatively. */
        s->dflash_serial_ms = dflash_timing_ema(s->dflash_serial_ms, refresh_ms);
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

    /* Arm the tap capture and the per-step snapshot. Everything from here to
     * dflash_cycle_disarm() below is instrumentation state that is global to
     * the process, so every exit in between has to go through the disarm --
     * an armed capture left behind would write the next serial decode's tap
     * rows into this cycle's buffer with this cycle's row count. */
    g_glm_dflash_cap.enabled = 1;
    g_glm_dflash_cap.buf = g->dflash_capture;
    g_glm_dflash_cap.n_rows = L;
    g_glm_dflash_cap.n_taps = n_taps;
    for (uint32_t j = 0; j < n_taps; j++) {
        g_glm_dflash_cap.taps[j] = dw->target_layers[j];
    }
    /* Last point before any target state is mutated: take the complete
     * speculative-state backup the rejection path will restore from.  Both
     * exits below happen before that mutation, so the session's checkpoint is
     * still exactly what it was and stays valid; only the instrumentation
     * needs undoing. */
    bool state_saved = false;
    if (dflash_rollback_needs_save(wide_state, allow_wide) || force_replay) {
        state_saved = glm53_graph_copy_spec_state_to(
                g, g->dflash_state_backup, true) &&
                !dflash_fail_injected(DS4_DFLASH_FAIL_STATE_SAVE);
        if (!state_saved) {
            dflash_cycle_disarm();
            snprintf(err, errlen, "dflash: speculative state save failed");
            return -1;
        }
    }
    if (!dflash_rollback_can_run(wide_state, allow_wide, state_saved,
                                 g->dflash_stepsnap != NULL)) {
        dflash_cycle_disarm();
        snprintf(err, errlen, "dflash: no sound speculative rollback");
        return -1;
    }
    if (dflash_fail_injected(DS4_DFLASH_FAIL_AFTER_ARM)) {
        dflash_cycle_disarm();
        snprintf(err, errlen, "dflash: injected failure after arm");
        return -1;
    }
    const double t_draft = getenv("DS4_DFLASH_STATS") ? now_sec() : 0.0;
    ds4_gpu_glm53_kda_stepsnap_begin(g->dflash_stepsnap,
                                     (uint32_t)(state_bytes / sizeof(float)),
                                     DS4_DFLASH2_MAX_DRAFT + 1u);
    ds4_gpu_glm53_tail_stepsnap_begin(g->dflash_stepsnap, state_bytes,
                                      kda_bytes, L, pos);
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
    const bool stepsnap_complete =
        ds4_gpu_glm53_kda_stepsnap_complete(kda_bytes) &&
        ds4_gpu_glm53_tail_stepsnap_complete();
    dflash_cycle_disarm();
    if (verified && !stepsnap_complete && getenv("DS4_DFLASH_STATS")) {
        fprintf(stderr, "ds4: dflash incomplete prefix capture; using replay\n");
    }
    const double t_verify = getenv("DS4_DFLASH_STATS") ? now_sec() : 0.0;
    if (!verified || dflash_fail_injected(DS4_DFLASH_FAIL_AFTER_VERIFY)) {
        dflash_cycle_fail_after_verify(s, g, state_saved);
        snprintf(err, errlen, "dflash: verify forward failed");
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
        dflash_cycle_fail_after_verify(s, g, state_saved);
        snprintf(err, errlen, "dflash: hc upload failed");
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
            dflash_cycle_fail_after_verify(s, g, state_saved);
            snprintf(err, errlen, "dflash: verify head failed");
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
    const ds4_dflash_rollback_plan plan =
        dflash_rollback_plan(L, n_committed, pos, wide_state, allow_wide,
                             state_saved, force_replay || !stepsnap_complete);
    bool replayed = false;
    switch (plan.kind) {
    case DS4_DFLASH_ROLLBACK_NONE:
        break;
    case DS4_DFLASH_ROLLBACK_STEPSNAP:
        if (!glm53_graph_restore_dflash_stepsnap(g, g->dflash_stepsnap,
                                              plan.stepsnap_index,
                                              state_bytes)) {
            snprintf(err, errlen, "dflash: prefix state restore failed");
            s->checkpoint_valid = false;
            return -1;
        }
        break;
    case DS4_DFLASH_ROLLBACK_REPLAY: {
        /* Undo the whole block, then walk the committed prefix forward
         * one token at a time exactly as serial decode would.  Every
         * per-step producer -- KDA conv/recurrent state, the indexer tail
         * ring, the positional caches -- is rebuilt by the replay, so the
         * frontier is the serial frontier and its logits (the last
         * forward's output) are the serial logits. */
        bool ok = glm53_graph_copy_spec_state_to(g, g->dflash_state_backup,
                                                 false);
        for (uint32_t i = 0; ok && i < plan.replay_tokens; i++) {
            ok = glm_graph_forward_token(g, &e->model, &e->weights, toks[i],
                                         NULL, pos + i, NULL, s->logits,
                                         false);
        }
        if (!ok) {
            snprintf(err, errlen, "dflash: speculative state replay failed");
            s->checkpoint_valid = false;
            return -1;
        }
        replayed = true;
        break;
    }
    case DS4_DFLASH_ROLLBACK_REFUSE:
    default:
        /* Unreachable: dflash_rollback_can_run() screened this before the
         * verify. Keep the state consistent rather than commit anything. */
        dflash_cycle_fail_after_verify(s, g, state_saved);
        snprintf(err, errlen, "dflash: no sound speculative rollback");
        return -1;
    }
    if (!replayed) {
        memcpy(s->logits, row_logits + (size_t)n_accept * DS4_N_VOCAB,
               (size_t)DS4_N_VOCAB * sizeof(float));
    }
    if (rejected_tok >= 0) s->logits[rejected_tok] = -1e30f;
    for (uint32_t i = 0; i < n_committed; i++) {
        token_vec_push(&s->checkpoint, toks[i]);
        accepted[i] = toks[i];
    }
    s->checkpoint_valid = true;
    if ((uint32_t)s->checkpoint.len != plan.frontier_pos) {
        /* The committed frontier must be exactly what the rollback plan
         * priced; anything else means the two disagree about how much of
         * the block survived. */
        snprintf(err, errlen,
                 "dflash: frontier %d does not match plan %u",
                 s->checkpoint.len, plan.frontier_pos);
        s->checkpoint_valid = false;
        return -1;
    }
    /* Compact KV is append-only at absolute positions. Compressed indexer
     * keys are visible only up to floor(frontier/4); scalar score_rows uses
     * visible/4 and batch scores mask per query. Entries produced solely by
     * rejected rows stay outside that range and are overwritten before use.
     * No independent GLM compact-cache live count is advanced by verify.
     * Dense-cache validity likewise advances only over committed rows. */
    ds4_session_glm_note_dense_cache(s, pos, n_committed);

    if (g_dflash_script.active) {
        /* Both arms record every committed token, so the comparison is over
         * token ids rather than rendered text. */
        for (uint32_t i = 0; i < n_committed; i++) {
            dflash_script_record_id(pos + i, toks[i]);
        }
    }
    if (sblk) {
        /* The certification line: what was asked for, and what the verifier
         * and the rollback actually did. */
        fprintf(stderr,
                "ds4: dflash script pos=%u k=%d n=%d drafted=%d accepted=%d "
                "committed=%u rollback=%s frontier=%u\n",
                pos, sblk->k, sblk->n, n_draft, n_accept, n_committed,
                plan.kind == DS4_DFLASH_ROLLBACK_NONE ? "none" :
                plan.kind == DS4_DFLASH_ROLLBACK_STEPSNAP ? "stepsnap" :
                plan.kind == DS4_DFLASH_ROLLBACK_REPLAY ? "replay" : "refuse",
                plan.frontier_pos);
        if (dflash_script_dump_frontier(plan.frontier_pos)) {
            dflash_script_dump_logits(plan.frontier_pos, s->logits);
            dflash_script_dump_state(plan.frontier_pos, g, "speculative");
        }
    }

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
                "ds4: dflash cycle drafted=%d accepted=%d rollback=%s "
                "ctx=%u draft=%.1fms verify=%.1fms heads=%.1fms tail=%.1fms "
                "serial=%.1fms cyc=%.1fms be=%.2f\n",
                n_draft, n_accept,
                plan.kind == DS4_DFLASH_ROLLBACK_NONE ? "none" :
                plan.kind == DS4_DFLASH_ROLLBACK_STEPSNAP ? "stepsnap" :
                plan.kind == DS4_DFLASH_ROLLBACK_REPLAY ? "replay" : "refuse",
                s->dflash_cache.n_layer ? s->dflash_cache.layer[0].len : 0u,
                (t_draft - t_cycle0) * 1000.0,
                (t_verify - t_draft) * 1000.0,
                (t_heads - t_verify) * 1000.0,
                (now_sec() - t_heads) * 1000.0,
                s->dflash_serial_ms, s->dflash_cycle_ms, breakeven);
    }
    {
        const float cyc_ms = (float)((now_sec() - cyc_t0) * 1000.0);
        s->dflash_cycle_ms = dflash_timing_ema(s->dflash_cycle_ms, cyc_ms);
        g_dflash_overhead.cycle_ms += (double)cyc_ms;
        g_dflash_overhead.cycles++;
        g_dflash_overhead.drafted += (uint64_t)n_draft;
        g_dflash_overhead.accepted += (uint64_t)n_accept;
        g_dflash_overhead.committed += n_committed;
        if (plan.kind == DS4_DFLASH_ROLLBACK_REPLAY) {
            g_dflash_overhead.replays++;
            g_dflash_overhead.replay_tokens += plan.replay_tokens;
        } else if (plan.kind == DS4_DFLASH_ROLLBACK_STEPSNAP) {
            g_dflash_overhead.stepsnaps++;
        }
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
