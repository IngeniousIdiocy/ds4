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
    g_glm_dflash_cap.fuse_sdn = 0;
    g_glm_dflash_cap.fuse_replace = 0;
    g_glm_dflash_cap.source_row = 0;
    g_glm_dflash_cap.captured_taps = 0;
    g_glm_dflash_cap.fused_taps = 0;
    g_glm_dflash_cap.buf = NULL;
    g_glm_dflash_cap.fused_buf = NULL;
    g_glm_dflash_cap.n_rows = 0;
    ds4_gpu_glm53_kda_rowsnap_end();
    ds4_gpu_glm53_tail_stepsnap_end();
}

/* Diagnostic only: compare every logit on identical real verifier inputs.
 * The caller keeps the scalar logits for acceptance even when differences
 * are observed. Margins distinguish a close decision from a large shift. */
static void dflash_head_compare_logits(uint32_t pos, uint32_t rows,
                                       const float *reference,
                                       const float *candidate) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *a = reference + (uint64_t)r * DS4_N_VOCAB;
        const float *b = candidate + (uint64_t)r * DS4_N_VOCAB;
        const int ai = glm_session_logits_argmax(a), bi = glm_session_logits_argmax(b);
        float a_second = -FLT_MAX, b_second = -FLT_MAX;
        double max_abs = 0;
        uint32_t different = 0, nonfinite = 0;
        for (uint32_t j = 0; j < DS4_N_VOCAB; j++) {
            uint32_t ab, bb;
            memcpy(&ab, &a[j], sizeof(ab));
            memcpy(&bb, &b[j], sizeof(bb));
            different += ab != bb;
            if ((ab & 0x7f800000u) == 0x7f800000u ||
                (bb & 0x7f800000u) == 0x7f800000u) nonfinite++;
            else {
                const double d = fabs((double)a[j] - (double)b[j]);
                if (d > max_abs) max_abs = d;
            }
            if ((int)j != ai && a[j] > a_second) a_second = a[j];
            if ((int)j != bi && b[j] > b_second) b_second = b[j];
        }
        fprintf(stderr, "ds4: dflash head compare pos=%u row=%u words=%u "
                "different=%u nonfinite=%u max_abs=%.9g ref_argmax=%d "
                "nt4_argmax=%d ref_margin=%.9g nt4_margin=%.9g\n",
                pos, r, (unsigned)DS4_N_VOCAB, different, nonfinite, max_abs,
                ai, bi, (double)a[ai] - a_second, (double)b[bi] - b_second);
    }
}

static bool dflash_verify_env_on(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

/* Shipped verifier paths default on; only "=0" disables one. */
static bool dflash_verify_default_on(const char *name) {
    const char *v = getenv(name);
    return !(v && !strcmp(v, "0"));
}

/* Run exactly one target verifier forward with process-global experimental
 * selectors scoped to that forward. Restore both before returning because
 * drafting uses the same BF16 selector. These opt-ins are discriminator
 * controls and require the existing single-active-request GPU schedule. */
static bool dflash_target_verify_forward(
        ds4_engine *e,
        ds4_glm_gpu_graph *g,
        const int *toks,
        uint32_t pos,
        uint32_t rows,
        float *hc_rows,
        bool bf16_mm,
        bool indexer_tiled) {
    const uint32_t old_bf16_mv_max = ds4_gpu_glm53_bf16_mv_max_get();
    const int old_indexer_tiled = ds4_gpu_glm53_indexer_small_tiled_get();
    if (bf16_mm && rows > 1u) ds4_gpu_glm53_bf16_mv_max_set(1u);
    ds4_gpu_glm53_indexer_small_tiled_set(indexer_tiled ? 1 : 0);

    bool verified;
    if (getenv("DS4_DFLASH_VERIFY_ROWS") &&
        glm53_graph_verify_rows_ready(g, pos, rows)) {
        /* A false return is a hard error: KDA state may already have advanced,
         * so the caller must restore rather than fall through to batch. */
        verified = glm53_graph_verify_rows(g, &e->model, &e->weights,
                                           toks, pos, rows, hc_rows, NULL);
    } else if (!glm53_graph_use_indexed_prefill(g) &&
               glm_graph_span_fits_full_attention(g, pos, rows)) {
        verified = glm_graph_forward_tokens(g, &e->model, &e->weights,
                                            toks, NULL, NULL, 0,
                                            pos, rows, hc_rows, NULL,
                                            NULL, NULL, pos, 0, rows);
    } else {
        verified = glm_graph_forward_indexed_tokens(g, &e->model, &e->weights,
                                                    toks, NULL, NULL, 0,
                                                    pos, rows, hc_rows, NULL,
                                                    NULL, NULL, pos, 0, rows);
    }

    ds4_gpu_glm53_indexer_small_tiled_set(old_indexer_tiled);
    ds4_gpu_glm53_bf16_mv_max_set(old_bf16_mv_max);
    return verified;
}

static void dflash_verify_rearm(ds4_glm_gpu_graph *g,
                                uint64_t state_bytes,
                                uint64_t kda_bytes,
                                uint32_t rows,
                                uint32_t pos) {
    g_glm_dflash_cap.source_row = 0;
    g_glm_dflash_cap.captured_taps = 0;
    ds4_gpu_glm53_kda_stepsnap_begin(g->dflash_stepsnap,
                                     (uint32_t)(state_bytes / sizeof(float)),
                                     DS4_DFLASH2_MAX_DRAFT + 1u);
    ds4_gpu_glm53_tail_stepsnap_begin(g->dflash_stepsnap, state_bytes,
                                      kda_bytes, rows, pos);
}

typedef struct {
    uint64_t words;
    uint64_t different;
    uint64_t nonfinite;
    double max_abs;
} dflash_verify_byte_diff;

static void dflash_verify_compare_words(const void *av,
                                        const void *bv,
                                        uint64_t bytes,
                                        dflash_verify_byte_diff *out) {
    const uint32_t *a = av;
    const uint32_t *b = bv;
    const uint64_t words = bytes / sizeof(uint32_t);
    for (uint64_t i = 0; i < words; i++) {
        out->words++;
        float af, bf;
        memcpy(&af, &a[i], sizeof(af));
        memcpy(&bf, &b[i], sizeof(bf));
        const bool finite = dflash_confidence_f32_bits_finite(a[i]) &&
                            dflash_confidence_f32_bits_finite(b[i]);
        if (!finite) out->nonfinite++;
        if (a[i] == b[i]) continue;
        out->different++;
        if (finite) {
            const double d = fabs((double)af - (double)bf);
            if (d > out->max_abs) out->max_abs = d;
        }
    }
}

static bool dflash_verify_compare_tensors(
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint64_t bytes,
        dflash_verify_byte_diff *out) {
    const uint64_t chunk_cap = 1u << 20;
    uint8_t *abuf = malloc((size_t)chunk_cap);
    uint8_t *bbuf = malloc((size_t)chunk_cap);
    if (!abuf || !bbuf) {
        free(abuf);
        free(bbuf);
        return false;
    }
    bool ok = true;
    for (uint64_t off = 0; ok && off < bytes; off += chunk_cap) {
        const uint64_t n = bytes - off < chunk_cap ? bytes - off : chunk_cap;
        ok = ds4_gpu_tensor_read(a, off, abuf, n) &&
             ds4_gpu_tensor_read(b, off, bbuf, n);
        if (ok) dflash_verify_compare_words(abuf, bbuf, n, out);
    }
    free(abuf);
    free(bbuf);
    return ok;
}

static void dflash_verify_compare_report(uint32_t pos,
                                         uint32_t rows,
                                         const char *pass,
                                         const char *item,
                                         const dflash_verify_byte_diff *d) {
    fprintf(stderr,
            "ds4: dflash verify compare pos=%u rows=%u pass=%s item=%s "
            "words=%llu different=%llu nonfinite=%llu max_abs=%.9g exact=%d\n",
            pos, rows, pass, item,
            (unsigned long long)d->words,
            (unsigned long long)d->different,
            (unsigned long long)d->nonfinite,
            d->max_abs, d->different == 0u);
}

static void dflash_capture_arm(const ds4_dflash2_weights *dw,
                               ds4_gpu_tensor *buf,
                               uint32_t rows,
                               uint32_t source_row,
                               bool fuse_sdn,
                               bool replace,
                               ds4_gpu_tensor *fused_buf) {
    g_glm_dflash_cap.source_row = source_row;
    g_glm_dflash_cap.captured_taps = 0;
    g_glm_dflash_cap.fused_taps = 0;
    g_glm_dflash_cap.enabled = 1;
    g_glm_dflash_cap.fuse_sdn = fuse_sdn;
    g_glm_dflash_cap.fuse_replace = replace;
    g_glm_dflash_cap.buf = buf;
    g_glm_dflash_cap.fused_buf = fused_buf;
    g_glm_dflash_cap.n_rows = rows;
    g_glm_dflash_cap.n_taps = dw->n_target;
    for (uint32_t j = 0; j < dw->n_target; j++)
        g_glm_dflash_cap.taps[j] = dw->target_layers[j];
}

/* Optional first-block microbenchmark. Same resident normalized rows, no
 * model forward or state mutation. Mean wall time includes encode+completion;
 * these diagnostic runs must never be quoted as generation throughput. */
static bool dflash_head_bench(ds4_session *s, uint32_t pos) {
    static bool done;
    const char *setting = getenv("DS4_DFLASH_HEAD_BENCH");
    if (!setting || done) return true;
    done = true;
    uint32_t count = (uint32_t)strtoul(setting, NULL, 10);
    if (count == 0) count = 8;
    if (count > 128) count = 128;
    ds4_engine *e = s->engine;
    for (uint32_t trial = 0; trial < 3; trial++) {
        for (uint32_t arm = 0; arm < 2; arm++) {
            const bool nt4 = ((arm + trial) % 2u) != 0;
            const uint64_t t0 = dflash_clock_now_ns();
            bool ok = ds4_gpu_begin_commands() != 0;
            for (uint32_t i = 0; ok && i < count; i++) {
                ok = nt4 ? ds4_gpu_dflash_head_nt4_tensor(
                    s->glm_graph.dflash_head_compare, e->model.map, e->model.size,
                    e->weights.output->abs_offset, DS4_N_EMBD, DS4_N_VOCAB,
                    g_dflash_pool.batch_norm, 8) :
                    ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                    s->glm_graph.dflash_head_compare, e->model.map, e->model.size,
                    e->weights.output->abs_offset, DS4_N_EMBD, DS4_N_VOCAB,
                    g_dflash_pool.batch_norm, 8);
            }
            if (!ds4_gpu_end_commands()) ok = false;
            fprintf(stderr, "ds4: dflash head bench pos=%u trial=%u arm=%s "
                    "iterations=%u rows=8 mean_wall_ms=%.6f ok=%d\n", pos,
                    trial + 1u, nt4 ? "nt4" : "scalar_rows", count,
                    dflash_clock_elapsed_ms(t0) / count, ok);
            if (!ok) return false;
        }
    }
    return true;
}

enum {
    DFLASH_CAPTURE_OFF = 0,
    DFLASH_CAPTURE_SEPARATE = 1,
    DFLASH_CAPTURE_FUSED = 2,
    DFLASH_CAPTURE_ARMS = 3,
};

static const char *dflash_capture_arm_name(uint32_t arm) {
    static const char *const names[DFLASH_CAPTURE_ARMS] = {
        "off", "separate", "fused"
    };
    return arm < DFLASH_CAPTURE_ARMS ? names[arm] : "invalid";
}

static void dflash_capture_compare_report(
        uint32_t pos, const char *arm, const char *item,
        const dflash_verify_byte_diff *d) {
    fprintf(stderr,
            "ds4: dflash capture compare pos=%u arm=%s item=%s "
            "words=%llu different=%llu nonfinite=%llu max_abs=%.9g exact=%d\n",
            pos, arm, item, (unsigned long long)d->words,
            (unsigned long long)d->different,
            (unsigned long long)d->nonfinite, d->max_abs,
            d->different == 0u);
}

/* Explicit diagnostic only.  Each arm starts from the same checkpoint and
 * complete recurrent/indexer-tail state, evaluates the same real token, and
 * retains its full interval.  Trial zero is marked cold and excluded only
 * from the separately reported warm mean.  The caller subsequently runs the
 * ordinary standalone-capture step, which is the result retained by the
 * session. */
static bool dflash_capture_compare_serial(ds4_session *s, int token,
                                          char *err, size_t errlen) {
    const char *compare = getenv("DS4_DFLASH_CAPTURE_COMPARE");
    const char *bench = getenv("DS4_DFLASH_CAPTURE_BENCH");
    if ((!compare || !compare[0] || !strcmp(compare, "0")) &&
        (!bench || !bench[0] || !strcmp(bench, "0"))) return true;
    static int claimed;
    if (!__sync_bool_compare_and_swap(&claimed, 0, 1)) return true;
    if (glm53_exact_mode_c()) {
        fprintf(stderr, "ds4: dflash capture compare skipped in exact mode\n");
        return true;
    }

    uint32_t trials = 1u;
    if (bench && bench[0] && strcmp(bench, "0")) {
        char *end = NULL;
        const unsigned long parsed = strtoul(bench, &end, 10);
        trials = end != bench && !*end && parsed ? (uint32_t)parsed : 4u;
    }
    if (trials > 16u) trials = 16u;

    ds4_glm_gpu_graph *g = &s->glm_graph;
    const ds4_dflash2_weights *dw = &s->engine->dflash_weights;
    const uint32_t pos = (uint32_t)s->checkpoint.len;
    const uint32_t dense_len = s->glm_dense_cache_len;
    const bool checkpoint_valid = s->checkpoint_valid;
    const uint64_t state_bytes = glm53_graph_spec_state_bytes(g);
    const uint64_t hc_bytes =
        (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    const uint64_t feature_bytes =
        (uint64_t)dw->n_target * DS4_N_EMBD * sizeof(float);
    if (state_bytes == 0u) {
        fprintf(stderr, "ds4: dflash capture compare has no serializable state; skipped\n");
        return true;
    }
    ds4_gpu_tensor *pre_state = state_bytes ? ds4_gpu_tensor_alloc(state_bytes) : NULL;
    ds4_gpu_tensor *arm_state[DFLASH_CAPTURE_ARMS] = {0};
    ds4_gpu_tensor *feature_gpu[2] = {0};
    float *arm_hc[DFLASH_CAPTURE_ARMS] = {0};
    float *arm_logits[DFLASH_CAPTURE_ARMS] = {0};
    float *feature_host[2] = {0};
    bool allocated = pre_state != NULL;
    bool safe = true;
    for (uint32_t arm = 0; arm < DFLASH_CAPTURE_ARMS; arm++) {
        arm_state[arm] = state_bytes ? ds4_gpu_tensor_alloc(state_bytes) : NULL;
        arm_hc[arm] = malloc((size_t)hc_bytes);
        arm_logits[arm] = malloc((size_t)logits_bytes);
        allocated = allocated && arm_state[arm] && arm_hc[arm] && arm_logits[arm];
    }
    for (uint32_t i = 0; i < 2; i++) {
        feature_gpu[i] = ds4_gpu_tensor_alloc(feature_bytes);
        feature_host[i] = malloc((size_t)feature_bytes);
        allocated = allocated && feature_gpu[i] && feature_host[i];
    }
    if (!allocated) {
        fprintf(stderr, "ds4: dflash capture compare allocation failed; skipped\n");
        goto cleanup;
    }
    if (!glm53_graph_copy_spec_state_to(g, pre_state, true)) {
        fprintf(stderr, "ds4: dflash capture compare state save failed; skipped\n");
        goto cleanup;
    }

    double all_ms[DFLASH_CAPTURE_ARMS] = {0};
    double warm_ms[DFLASH_CAPTURE_ARMS] = {0};
    uint32_t all_n[DFLASH_CAPTURE_ARMS] = {0};
    uint32_t warm_n[DFLASH_CAPTURE_ARMS] = {0};
    dflash_verify_byte_diff hc_diff[2] = {0};
    dflash_verify_byte_diff logits_diff[2] = {0};
    dflash_verify_byte_diff state_diff[2] = {0};
    dflash_verify_byte_diff feature_diff = {0};
    bool ran = true;
    for (uint32_t trial = 0; ran && trial < trials; trial++) {
        uint32_t capture_mask[DFLASH_CAPTURE_ARMS] = {0};
        uint32_t fused_mask[DFLASH_CAPTURE_ARMS] = {0};
        for (uint32_t k = 0; ran && k < DFLASH_CAPTURE_ARMS; k++) {
            const uint32_t arm = (trial + k) % DFLASH_CAPTURE_ARMS;
            dflash_cycle_disarm();
            ran = glm53_graph_copy_spec_state_to(g, pre_state, false);
            s->checkpoint.len = (int)pos;
            s->glm_dense_cache_len = dense_len;
            s->checkpoint_valid = checkpoint_valid;
            if (!ran) break;
            if (arm == DFLASH_CAPTURE_SEPARATE) {
                dflash_capture_arm(dw, feature_gpu[0], 1u, 0u,
                                   false, false, NULL);
            } else if (arm == DFLASH_CAPTURE_FUSED) {
                dflash_capture_arm(dw, feature_gpu[1], 1u, 0u,
                                   true, true, NULL);
            }
            char diag_err[256] = {0};
            const double t0 = now_sec();
            const int rc = ds4_session_eval_internal(
                s, token, false, diag_err, sizeof(diag_err));
            const double elapsed_ms = (now_sec() - t0) * 1000.0;
            capture_mask[arm] = g_glm_dflash_cap.captured_taps;
            fused_mask[arm] = g_glm_dflash_cap.fused_taps;
            dflash_cycle_disarm();
            fprintf(stderr,
                    "ds4: dflash capture timing pos=%u trial=%u cold=%d "
                    "arm=%s ms=%.6f status=%s capture=0x%x fused=0x%x\n",
                    pos, trial, trial == 0u, dflash_capture_arm_name(arm),
                    elapsed_ms, rc == 0 ? "ok" : "failed",
                    capture_mask[arm], fused_mask[arm]);
            all_ms[arm] += elapsed_ms;
            all_n[arm]++;
            if (trial != 0u) {
                warm_ms[arm] += elapsed_ms;
                warm_n[arm]++;
            }
            ran = rc == 0 &&
                glm53_graph_copy_spec_state_to(g, arm_state[arm], true) &&
                ds4_gpu_tensor_read(g->hc_cur, 0, arm_hc[arm], hc_bytes);
            if (ran) memcpy(arm_logits[arm], s->logits, (size_t)logits_bytes);
            if (ran && arm == DFLASH_CAPTURE_SEPARATE)
                ran = ds4_gpu_tensor_read(feature_gpu[0], 0,
                                           feature_host[0], feature_bytes);
            if (ran && arm == DFLASH_CAPTURE_FUSED)
                ran = ds4_gpu_tensor_read(feature_gpu[1], 0,
                                           feature_host[1], feature_bytes);
        }
        if (!ran) break;
        const uint32_t expected = dw->n_target < 32u ?
            (1u << dw->n_target) - 1u : UINT32_MAX;
        ran = capture_mask[DFLASH_CAPTURE_SEPARATE] == expected &&
              capture_mask[DFLASH_CAPTURE_FUSED] == expected &&
              fused_mask[DFLASH_CAPTURE_FUSED] == expected;
        for (uint32_t arm = DFLASH_CAPTURE_SEPARATE;
             arm < DFLASH_CAPTURE_ARMS; arm++) {
            const uint32_t i = arm - 1u;
            dflash_verify_compare_words(arm_hc[DFLASH_CAPTURE_OFF],
                                        arm_hc[arm], hc_bytes, &hc_diff[i]);
            dflash_verify_compare_words(arm_logits[DFLASH_CAPTURE_OFF],
                                        arm_logits[arm], logits_bytes,
                                        &logits_diff[i]);
            ran = ran && dflash_verify_compare_tensors(
                arm_state[DFLASH_CAPTURE_OFF], arm_state[arm],
                state_bytes, &state_diff[i]);
        }
        dflash_verify_compare_words(feature_host[0], feature_host[1],
                                    feature_bytes, &feature_diff);
    }

    for (uint32_t arm = 0; arm < DFLASH_CAPTURE_ARMS; arm++) {
        fprintf(stderr,
                "ds4: dflash capture timing summary pos=%u arm=%s "
                "all_n=%u all_mean_ms=%.6f warm_n=%u warm_mean_ms=%.6f\n",
                pos, dflash_capture_arm_name(arm), all_n[arm],
                all_n[arm] ? all_ms[arm] / all_n[arm] : 0.0,
                warm_n[arm], warm_n[arm] ? warm_ms[arm] / warm_n[arm] : 0.0);
    }
    for (uint32_t arm = DFLASH_CAPTURE_SEPARATE;
         arm < DFLASH_CAPTURE_ARMS; arm++) {
        const uint32_t i = arm - 1u;
        dflash_capture_compare_report(pos, dflash_capture_arm_name(arm),
                                      "hc", &hc_diff[i]);
        dflash_capture_compare_report(pos, dflash_capture_arm_name(arm),
                                      "logits", &logits_diff[i]);
        dflash_capture_compare_report(pos, dflash_capture_arm_name(arm),
                                      "state", &state_diff[i]);
    }
    dflash_capture_compare_report(pos, "fused", "features", &feature_diff);
    if (!ran)
        fprintf(stderr, "ds4: dflash capture compare incomplete or missing taps\n");

    dflash_cycle_disarm();
    if (!glm53_graph_copy_spec_state_to(g, pre_state, false)) {
        snprintf(err, errlen,
                 "dflash: capture comparator could not restore target state");
        safe = false;
    }
    s->checkpoint.len = (int)pos;
    s->glm_dense_cache_len = dense_len;
    s->checkpoint_valid = checkpoint_valid;

cleanup:
    dflash_cycle_disarm();
    ds4_gpu_tensor_free(pre_state);
    for (uint32_t arm = 0; arm < DFLASH_CAPTURE_ARMS; arm++) {
        ds4_gpu_tensor_free(arm_state[arm]);
        free(arm_hc[arm]);
        free(arm_logits[arm]);
    }
    for (uint32_t i = 0; i < 2; i++) {
        ds4_gpu_tensor_free(feature_gpu[i]);
        free(feature_host[i]);
    }
    return safe;
}

/* A latched drafter failure never arms capture or touches the drafter again.
 * A failed drain is different: request failure is safer than encoding target
 * work against command state whose completion could not be established. */
static int ds4_session_glm_dflash_fault_serial(ds4_session *s, int first_token,
        int *accepted, char *err, size_t errlen) {
    ds4_dflash_adaptive_trace *trace = &s->dflash_trace;
    trace->operation = "draft-disabled";
    if (!dflash_fault_skip(&s->dflash_fault)) {
        s->checkpoint_valid = false;
        snprintf(err, errlen, "dflash: drafter command drain failed; session cannot resume");
        return -1;
    }
    const uint64_t start = dflash_clock_now_ns();
    const int rc = ds4_session_eval_internal(s, first_token, false, err, errlen);
    trace->serial_ms = dflash_clock_elapsed_ms(start);
    if (rc != 0) return -1;
    accepted[0] = first_token;
    return 1;
}

static void ds4_session_dflash_history_stage(ds4_session *s, uint32_t rows, uint32_t end) {
    if (s->dflash_full_block ||
        (!s->dflash_adaptive.config.retry && !s->dflash_adaptive.config.min_serial_tokens)) {
        s->dflash_pending_rows = rows;
        s->dflash_pending_end_pos = end;
        return;
    }
    const size_t width = (size_t)s->engine->dflash_weights.n_target * DS4_N_EMBD;
    if (!dflash_history_append(&s->dflash_history, s->dflash_history_rows,
            s->dflash_pending, rows, width, end)) s->dflash_history.len = 0;
    s->dflash_pending_rows = s->dflash_history.len;
    s->dflash_pending_end_pos = s->dflash_history.end;
}

static int ds4_session_glm_dflash_capture_serial(ds4_session *s, int first_token,
        int *accepted, char *err, size_t errlen) {
    ds4_glm_gpu_graph *g = &s->glm_graph;
    const ds4_dflash2_weights *dw = &s->engine->dflash_weights;
    const uint32_t pos = (uint32_t)s->checkpoint.len;
    const uint32_t n_taps = dw->n_target, H = DS4_N_EMBD;
    ds4_dflash_adaptive_trace *trace = &s->dflash_trace;
    if (!trace->operation) trace->operation = "refresh";
    if (!dflash_capture_compare_serial(s, first_token, err, errlen)) return -1;
    const char *capture_compare = getenv("DS4_DFLASH_CAPTURE_COMPARE");
    const char *capture_bench = getenv("DS4_DFLASH_CAPTURE_BENCH");
    const bool capture_diagnostic =
        (capture_compare && capture_compare[0] && strcmp(capture_compare, "0")) ||
        (capture_bench && capture_bench[0] && strcmp(capture_bench, "0"));
    /* Diagnostic runs always retain the ordinary standalone reference. */
    const bool fuse_sdn = !capture_diagnostic &&
                          glm_dflash_capture_sdn_requested();
    dflash_capture_arm(dw, g->dflash_capture, 1u, 0u,
                       fuse_sdn, fuse_sdn, NULL);
    const uint64_t refresh_t0 = dflash_clock_now_ns();
    const int rc = ds4_session_eval_internal(s, first_token, false,
                                             err, errlen);
    dflash_cycle_disarm();
    const uint64_t refresh_end = dflash_clock_now_ns();
    const double refresh_ms = dflash_clock_ms(refresh_t0, refresh_end);
    g_dflash_overhead.refresh_ms += refresh_ms;
    g_dflash_overhead.refreshes++;
    trace->serial_ms = refresh_ms;
    if (rc != 0) return -1;
    /* This is a measured serial target step even when no park has yet
     * occurred. Capture overhead stays charged conservatively. */
    s->dflash_serial_ms = dflash_timing_ema(s->dflash_serial_ms, refresh_ms);
    if (ds4_gpu_tensor_read(g->dflash_capture, 0, s->dflash_pending,
                            (uint64_t)n_taps * H * sizeof(float))) {
        ds4_session_dflash_history_stage(s, 1u, pos + 1u);
    } else {
        s->dflash_pending_rows = 0;
        s->dflash_history.len = 0;
    }
    trace->tail_ms = dflash_clock_elapsed_ms(refresh_end);
    accepted[0] = first_token;
    return 1;
}

static int ds4_session_glm_dflash_cycle_impl(
        ds4_session *s, int first_token,
        float temperature, int top_k, float top_p, float min_p,
        uint64_t *rng,
        bool ignore_eos, ds4_think_mode think_mode,
        int *accepted, int accepted_cap,
        char *err, size_t errlen) {
    ds4_engine *e = s->engine;
    ds4_glm_gpu_graph *g = &s->glm_graph;
    const ds4_dflash2_weights *dw = &e->dflash_weights;
    const uint32_t pos = (uint32_t)s->checkpoint.len;
    const uint32_t H = DS4_N_EMBD;
    const uint32_t n_taps = dw->n_target;
    /* Public speculative always uses the full block; conservative alone
     * applies confidence, width economics, retry, and the soft meter. */
    const bool controlled = !s->dflash_full_block;
    ds4_dflash_adaptive_trace *trace = &s->dflash_trace;
    const uint64_t call_start = dflash_clock_now_ns();
    if (s->dflash_fault.disabled)
        return ds4_session_glm_dflash_fault_serial(s, first_token, accepted, err, errlen);

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

    /* Capture is small and required by the exact serial refresh. Verifier
     * snapshots are allocated only after a nonempty confident prefix exists. */
    const uint64_t state_bytes = spec_bytes;
    const bool want_backup = wide_state || force_replay;
    if (!g->dflash_capture) {
        g->dflash_capture = ds4_gpu_tensor_alloc(
                (uint64_t)(DS4_DFLASH2_MAX_DRAFT + 1u) * n_taps * H * sizeof(float));
    }
    if (!g->dflash_capture) {
        snprintf(err, errlen, "dflash: capture alloc failed");
        return -1;
    }
    if (s->dflash_cache.n_layer == 0) {
        /* Preserve the existing supported context cap. Larger histories
         * require their own ingestion and end-to-end measurements. */
        const uint32_t ccap = glm_dflash_context_capacity(dw);
        if (!dflash_cache_init(&s->dflash_cache, dw, ccap)) {
            snprintf(err, errlen, "dflash: cache alloc failed");
            return -1;
        }
    }
    const uint32_t history_cap = controlled &&
        (s->dflash_adaptive.config.retry || s->dflash_adaptive.config.min_serial_tokens) ?
        s->dflash_cache.layer[0].cap : 0u;
    const uint32_t pending_cap = history_cap > DS4_DFLASH2_MAX_DRAFT + 1u ?
        history_cap : DS4_DFLASH2_MAX_DRAFT + 1u;
    if (s->dflash_pending_capacity < pending_cap) {
        s->dflash_pending = xrealloc(s->dflash_pending,
                (size_t)pending_cap * n_taps * H * sizeof(float));
        s->dflash_pending_capacity = pending_cap;
    }
    if (history_cap && !s->dflash_history_rows) {
        s->dflash_history_rows = xmalloc((size_t)history_cap * n_taps * H * sizeof(float));
        s->dflash_history.cap = history_cap;
    }
    if (controlled && (s->dflash_adaptive.reasoning ||
                      dflash_adaptive_entry_wait(&s->dflash_adaptive))) {
        /* Keep the recent prompt tail followed by every real entry row. A
         * delayed first proposal must not silently lose prompt conditioning. */
        if (!s->dflash_warm) {
            if (dflash_seed_ring_usable(g_glm_dflash_seed.owner_gen, s->dflash_gen,
                    g_glm_dflash_seed.ring_len, g_glm_dflash_seed.ring_end_pos, pos)) {
                trace->bootstrap_rows = g_glm_dflash_seed.ring_len;
                if (!dflash_history_append(&s->dflash_history, s->dflash_history_rows,
                        g_glm_dflash_seed.ring, g_glm_dflash_seed.ring_len,
                        (size_t)n_taps * H, pos)) s->dflash_history.len = 0;
                glm_dflash_seed_invalidate();
            }
            s->dflash_warm = 1;
        }
        trace->operation = s->dflash_adaptive.reasoning ? "reasoning-serial" : "entry-serial";
        trace->entry = !s->dflash_adaptive.reasoning;
        trace->setup_ms = dflash_clock_elapsed_ms(call_start);
        return ds4_session_glm_dflash_capture_serial(s, first_token, accepted, err, errlen);
    }
    if (controlled && s->dflash_adaptive.config.retry && s->dflash_adaptive.need_serial) {
        trace->operation = "serial-calibration";
        trace->serial_calibration = true;
        trace->setup_ms = dflash_clock_elapsed_ms(call_start);
        return ds4_session_glm_dflash_capture_serial(s, first_token, accepted, err, errlen);
    }
    const double funding_setup_ms = dflash_clock_elapsed_ms(call_start);
    const bool funded_retry = controlled &&
        dflash_adaptive_funded_retry(&s->dflash_adaptive, funding_setup_ms);
    if (controlled && s->dflash_adaptive.config.retry &&
            s->dflash_adaptive.skip_remaining && !funded_retry) {
        trace->operation = "retry-skip";
        trace->skipped = true;
        trace->setup_ms = dflash_clock_elapsed_ms(call_start);
        return ds4_session_glm_dflash_capture_serial(s, first_token, accepted, err, errlen);
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
            const bool fuse_sdn = glm_dflash_capture_sdn_requested();
            dflash_capture_arm(dw, g->dflash_capture, 1u, 0u,
                               fuse_sdn, fuse_sdn, NULL);
            const int rc = ds4_session_eval_internal(s, first_token, false,
                                                     err, errlen);
            dflash_cycle_disarm();
            if (rc != 0) return -1;
            if (ds4_gpu_tensor_read(g->dflash_capture, 0, s->dflash_pending,
                                    (uint64_t)n_taps * H * sizeof(float))) {
                ds4_session_dflash_history_stage(s, 1u, pos + 1u);
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

    /* Use a usable prompt tail directly in the first real proposal. A
     * throwaway seed proposal would pay for two draft passes at bootstrap.
     * Cache positions and pending end positions are explicit so interrupted
     * capture, session reuse or a rewind cannot silently join different
     * causal prefixes. */
    const uint64_t t_cycle0 = call_start;
    const uint64_t cyc_t0 = call_start;
    int drafts[DS4_DFLASH2_MAX_DRAFT];
    int n_draft = 0;
    if (s->dflash_history.len)
        dflash_history_read(&s->dflash_history, s->dflash_history_rows,
                s->dflash_pending, (size_t)n_taps * H);
    const float *conditioning = s->dflash_pending;
    uint32_t conditioning_rows = s->dflash_pending_rows;
    bool seed_used = false;
    if (conditioning_rows && s->dflash_pending_end_pos != pos) {
        conditioning_rows = s->dflash_pending_rows = 0;
        for (uint32_t i = 0; i < s->dflash_cache.n_layer; i++)
            s->dflash_cache.layer[i].len = 0;
        s->dflash_cache_end_pos = 0;
    }
    if (!s->dflash_warm) {
        if (dflash_seed_ring_usable(g_glm_dflash_seed.owner_gen, s->dflash_gen,
                g_glm_dflash_seed.ring_len, g_glm_dflash_seed.ring_end_pos, pos)) {
            conditioning = g_glm_dflash_seed.ring;
            conditioning_rows = g_glm_dflash_seed.ring_len;
            trace->bootstrap_rows = conditioning_rows;
            seed_used = true;
        }
        s->dflash_warm = 1;
    }
    if (conditioning_rows && !dflash_adaptive_context_contiguous(pos,
            conditioning_rows, seed_used ? pos : s->dflash_pending_end_pos,
            s->dflash_cache_end_pos)) {
        for (uint32_t i = 0; i < s->dflash_cache.n_layer; i++)
            s->dflash_cache.layer[i].len = 0;
        s->dflash_cache_end_pos = 0;
        if (conditioning_rows > pos) conditioning_rows = 0;
    }
    const float breakeven = s->dflash_serial_ms > 0.0f && s->dflash_cycle_ms > 0.0f ?
        s->dflash_cycle_ms / s->dflash_serial_ms : 0.0f;
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
    trace->setup_ms = dflash_clock_elapsed_ms(call_start);
    const uint64_t proposal_start = dflash_clock_now_ns();
    if (sblk) {
        n_draft = dflash_script_drafts(sblk, pos, drafts, DS4_N_VOCAB);
        s->dflash_pending_rows = 0;
        s->dflash_history.len = 0;
    } else if (conditioning_rows && l_clamp > 1u) {
        /* Block diffusion is trained jointly. Confidence/adaptation select
         * the verified prefix without changing the drafter's noise block. */
        const uint32_t model_drafts = dw->block_size - 1u < DS4_DFLASH2_MAX_DRAFT ?
            dw->block_size - 1u : DS4_DFLASH2_MAX_DRAFT;
        trace->recovery = controlled && s->dflash_adaptive.recovery_pending;
        trace->funded = funded_retry;
        if (funded_retry) {
            trace->funding_bank_ms = dflash_adaptive_savings_ms(&s->dflash_adaptive) - funding_setup_ms;
            trace->funding_cost_ms = s->dflash_adaptive.draft_estimate_ms;
        }
        trace->chosen = controlled ? model_drafts : l_clamp - 1u;
        float confidence[DS4_DFLASH2_MAX_DRAFT];
        for (uint32_t i = 0; i < DS4_DFLASH2_MAX_DRAFT; i++) confidence[i] = -1.0f;
        /* The session latch is consulted above before any setup/capture. */
        (void)dflash_fault_attempt(&s->dflash_fault);
        n_draft = dflash2_propose_confident(drafts, (int)trace->chosen,
                &e->dflash_model, dw, &e->model, &e->weights, conditioning,
                conditioning_rows, &s->dflash_cache, first_token, pos,
                controlled ? confidence : NULL, controlled);
        s->dflash_pending_rows = 0;
        s->dflash_history.len = 0;
        if (n_draft < 0) {
            /* Only drafter/shared scratch work ran: the target checkpoint,
             * logits and recurrent state still name the original frontier.
             * Drain before choosing serial fallback; never retry this session. */
            const bool drained = ds4_gpu_synchronize() != 0;
            dflash_fault_latch(&s->dflash_fault, drained);
            dflash_cycle_disarm();
            for (uint32_t i = 0; i < s->dflash_cache.n_layer; i++)
                s->dflash_cache.layer[i].len = 0;
            s->dflash_cache_end_pos = 0;
            s->dflash_pending_end_pos = 0;
            g_glm_dflash_seed.enabled = 0;
            glm_dflash_seed_invalidate();
            ds4_session_dflash_fault_report(s, "latched");
            trace->draft_ms = dflash_clock_elapsed_ms(proposal_start);
            return ds4_session_glm_dflash_fault_serial(s, first_token, accepted, err, errlen);
        } else {
            s->dflash_cache_end_pos = pos;
            trace->proposed = (uint32_t)n_draft;
            if (controlled) {
                uint32_t limit = dflash_adaptive_limit(&s->dflash_adaptive);
                if (limit > l_clamp - 1u) limit = l_clamp - 1u;
                if (n_draft) {
                    trace->confidence_first = confidence[0];
                    trace->confidence_min = confidence[0];
                    for (int i = 1; i < n_draft; i++)
                        if (confidence[i] < trace->confidence_min)
                            trace->confidence_min = confidence[i];
                }
                n_draft = (int)dflash_adaptive_prefix(confidence, (uint32_t)n_draft,
                    limit, s->dflash_adaptive.config.p_min,
                    s->dflash_adaptive.config.n_min);
                /* Admit on confidence, but verify the efficient eight-row
                 * target shape (default; DS4_DFLASH_VERIFY_FULL_BLOCK=0 verifies
                 * only the admitted prefix). The remaining real proposals are
                 * still target-verified; this neither fabricates padding tokens
                 * nor bypasses rejection. Context/output limits and developer
                 * width caps remain binding. */
                if (n_draft && dflash_verify_default_on("DS4_DFLASH_VERIFY_FULL_BLOCK")) {
                    n_draft = (int)(trace->proposed < l_clamp - 1u ?
                        trace->proposed : l_clamp - 1u);
                    if ((uint32_t)n_draft > s->dflash_adaptive.config.n_max)
                        n_draft = (int)s->dflash_adaptive.config.n_max;
                }
            }
        }
    }
    if (seed_used) glm_dflash_seed_invalidate();
    /* No proposer ran on a cold refresh: its cost is exactly zero. */
    trace->draft_ms = (sblk || trace->chosen) ?
        dflash_clock_elapsed_ms(proposal_start) : 0.0;
    trace->candidate = (uint32_t)n_draft;
    const uint32_t full_width = trace->chosen < s->dflash_adaptive.config.n_max ?
        trace->chosen : s->dflash_adaptive.config.n_max;
    /* One positive-evidence early check may measure an unseen full width.
     * Do not let an extrapolation suppress that sole bounded calibration. */
    const bool recovery_calibration = controlled && n_draft > 0 &&
        s->dflash_adaptive.recovery_pending && (uint32_t)n_draft == full_width &&
        s->dflash_adaptive.target_ms[n_draft] <= 0.0;
    if (recovery_calibration) trace->calibration = true;
    if (controlled && n_draft &&
            (((s->dflash_adaptive.probe_only || s->dflash_adaptive.recovery_pending) &&
              (uint32_t)n_draft < full_width) ||
             (!recovery_calibration && !dflash_adaptive_economic(&s->dflash_adaptive,
                (uint32_t)n_draft, &trace->calibration)))) {
        trace->economic = true;
        trace->operation = "economic-decline";
        n_draft = 0;
    }
    if (n_draft == 0)
        return ds4_session_glm_dflash_capture_serial(s, first_token, accepted, err, errlen);
    trace->operation = "verify";
    trace->verified = (uint32_t)n_draft;
    const uint64_t verify_setup_start = dflash_clock_now_ns();
    if (!g->dflash_stepsnap && state_bytes)
        g->dflash_stepsnap = ds4_gpu_tensor_alloc(
                (uint64_t)(DS4_DFLASH2_MAX_DRAFT + 1u) * state_bytes);
    if (want_backup && !g->dflash_state_backup && spec_bytes)
        g->dflash_state_backup = ds4_gpu_tensor_alloc(spec_bytes);
    if ((state_bytes && !g->dflash_stepsnap) ||
        (want_backup && !g->dflash_state_backup)) {
        snprintf(err, errlen, "dflash: verifier scratch alloc failed");
        return -1;
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
    dflash_capture_arm(dw, g->dflash_capture, L, 0u,
                       false, false, NULL);
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
    const uint64_t t_draft = dflash_clock_now_ns();
    trace->setup_ms += dflash_clock_ms(verify_setup_start, t_draft);
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
    const bool verify_bf16_mm_requested =
        dflash_verify_env_on("DS4_DFLASH_VERIFY_BF16_MM");
    const bool verify_indexer_tiled_requested =
        dflash_verify_env_on("DS4_DFLASH_VERIFY_INDEXER_TILED");
    const bool verify_candidate = !glm53_exact_mode_c() &&
        (verify_bf16_mm_requested || verify_indexer_tiled_requested);
    const bool verify_compare_requested =
        dflash_verify_env_on("DS4_DFLASH_VERIFY_KERNEL_COMPARE");

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    /* Diagnostic comparator: reference -> candidate -> reference. The last
     * reference is the retained verifier operation. It rewrites positional
     * caches at the same slots and restores the default HC rows, captured tap
     * features, prefix snapshots, and complete recurrent/tail state. */
    static int verify_compare_claimed;
    bool verify_compare = verify_compare_requested &&
        !glm53_exact_mode_c() &&
        state_saved &&
        (verify_bf16_mm_requested || verify_indexer_tiled_requested) &&
        __sync_bool_compare_and_swap(&verify_compare_claimed, 0, 1);
    const uint64_t verify_hc_bytes =
        (uint64_t)L * DS4_N_HC * H * sizeof(float);
    const uint64_t verify_capture_bytes =
        (uint64_t)L * n_taps * H * sizeof(float);
    ds4_gpu_tensor *verify_ref_state = NULL;
    ds4_gpu_tensor *verify_other_state = NULL;
    float *verify_ref_hc = NULL;
    float *verify_ref_capture = NULL;
    float *verify_other_capture = NULL;
    if (verify_compare) {
        verify_ref_state = ds4_gpu_tensor_alloc(spec_bytes);
        verify_other_state = ds4_gpu_tensor_alloc(spec_bytes);
        verify_ref_hc = malloc((size_t)verify_hc_bytes);
        verify_ref_capture = malloc((size_t)verify_capture_bytes);
        verify_other_capture = malloc((size_t)verify_capture_bytes);
        if (!verify_ref_state || !verify_other_state || !verify_ref_hc ||
            !verify_ref_capture || !verify_other_capture) {
            fprintf(stderr,
                    "ds4: dflash verifier comparator setup failed; "
                    "running one retained verifier\n");
            ds4_gpu_tensor_free(verify_ref_state);
            ds4_gpu_tensor_free(verify_other_state);
            free(verify_ref_hc);
            free(verify_ref_capture);
            free(verify_other_capture);
            verify_ref_state = verify_other_state = NULL;
            verify_ref_hc = verify_ref_capture = verify_other_capture = NULL;
            verify_compare = false;
        }
    }

    /* MODE 1 only: isolate the first verifier's dispatch census from prompt
     * prefill, target refresh, and drafting.  The atomic claim prevents two
     * server sessions from resetting the process-global ledger concurrently.
     * The three-forward comparator suppresses the census rather than mixing
     * diagnostic candidate and retained-reference dispatches. */
    static int verify_ledger_claimed;
    const char *verify_ledger_path = getenv("DS4_DFLASH_VERIFY_LEDGER");
    const bool verify_ledger = !verify_compare &&
        verify_ledger_path && verify_ledger_path[0] &&
        __sync_bool_compare_and_swap(&verify_ledger_claimed, 0, 1) &&
        ds4_gpu_kernel_ledger_window_begin();
#endif
    /* Experimental, default OFF: measured byte-identical to the batch
     * path but NOT faster (~105 vs ~107ms) — once the batch path gained
     * flush-every-8-layers encode overlap, the remaining verify cost is
     * genuine GPU kernel time, which this pass shares. Kept as the
     * skeleton for a future fused-rows-kernel push (hc_pre norm+mix rows
     * fusion, KDA prefill occupancy); also limited to pos+n <= 4096. */
    bool verified;
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (verify_compare) {
        const uint64_t ref1_t0 = dflash_clock_now_ns();
        const bool ref1_forward = dflash_target_verify_forward(
            e, g, toks, pos, L, hc_rows_host, false, false);
        const double ref1_ms = dflash_clock_elapsed_ms(ref1_t0);
        const uint32_t ref_capture_mask = g_glm_dflash_cap.captured_taps;
        bool ref1_saved = ref1_forward &&
            glm53_graph_copy_spec_state_to(g, verify_ref_state, true) &&
            ds4_gpu_tensor_read(g->dflash_capture, 0, verify_ref_capture,
                                verify_capture_bytes);
        if (ref1_saved) memcpy(verify_ref_hc, hc_rows_host,
                               (size_t)verify_hc_bytes);

        bool restored = ref1_saved && glm53_graph_copy_spec_state_to(
            g, g->dflash_state_backup, false);
        bool candidate_forward = false;
        bool candidate_saved = false;
        bool candidate_compared = false;
        double candidate_ms = 0.0;
        uint32_t candidate_capture_mask = 0;
        if (restored) {
            dflash_verify_rearm(g, state_bytes, kda_bytes, L, pos);
            const uint64_t candidate_t0 = dflash_clock_now_ns();
            candidate_forward = dflash_target_verify_forward(
                e, g, toks, pos, L, hc_rows_host,
                verify_bf16_mm_requested,
                verify_indexer_tiled_requested);
            candidate_ms = dflash_clock_elapsed_ms(candidate_t0);
            candidate_capture_mask = g_glm_dflash_cap.captured_taps;
            candidate_saved = candidate_forward &&
                glm53_graph_copy_spec_state_to(g, verify_other_state, true) &&
                ds4_gpu_tensor_read(g->dflash_capture, 0,
                                    verify_other_capture,
                                    verify_capture_bytes);
        }

        if (ref1_saved && candidate_saved) {
            dflash_verify_byte_diff hc_diff = {0};
            dflash_verify_byte_diff capture_diff = {0};
            dflash_verify_byte_diff state_diff = {0};
            dflash_verify_compare_words(verify_ref_hc, hc_rows_host,
                                        verify_hc_bytes, &hc_diff);
            dflash_verify_compare_words(verify_ref_capture,
                                        verify_other_capture,
                                        verify_capture_bytes, &capture_diff);
            const bool state_read = dflash_verify_compare_tensors(
                verify_ref_state, verify_other_state, spec_bytes, &state_diff);
            dflash_verify_compare_report(pos, L, "candidate", "hc", &hc_diff);
            dflash_verify_compare_report(pos, L, "candidate", "features",
                                         &capture_diff);
            if (state_read) {
                dflash_verify_compare_report(pos, L, "candidate", "state",
                                             &state_diff);
            }
            candidate_compared = state_read;
        }

        restored = restored && glm53_graph_copy_spec_state_to(
            g, g->dflash_state_backup, false);
        bool ref2_forward = false;
        bool ref2_saved = false;
        bool ref2_compared = false;
        double ref2_ms = 0.0;
        uint32_t ref2_capture_mask = 0;
        if (restored) {
            dflash_verify_rearm(g, state_bytes, kda_bytes, L, pos);
            const uint64_t ref2_t0 = dflash_clock_now_ns();
            ref2_forward = dflash_target_verify_forward(
                e, g, toks, pos, L, hc_rows_host, false, false);
            ref2_ms = dflash_clock_elapsed_ms(ref2_t0);
            ref2_capture_mask = g_glm_dflash_cap.captured_taps;
            ref2_saved = ref2_forward &&
                glm53_graph_copy_spec_state_to(g, verify_other_state, true) &&
                ds4_gpu_tensor_read(g->dflash_capture, 0,
                                    verify_other_capture,
                                    verify_capture_bytes);
        }
        if (ref1_saved && ref2_saved) {
            dflash_verify_byte_diff hc_diff = {0};
            dflash_verify_byte_diff capture_diff = {0};
            dflash_verify_byte_diff state_diff = {0};
            dflash_verify_compare_words(verify_ref_hc, hc_rows_host,
                                        verify_hc_bytes, &hc_diff);
            dflash_verify_compare_words(verify_ref_capture,
                                        verify_other_capture,
                                        verify_capture_bytes, &capture_diff);
            const bool state_read = dflash_verify_compare_tensors(
                verify_ref_state, verify_other_state, spec_bytes, &state_diff);
            dflash_verify_compare_report(pos, L, "replay", "hc", &hc_diff);
            dflash_verify_compare_report(pos, L, "replay", "features",
                                         &capture_diff);
            if (state_read) {
                dflash_verify_compare_report(pos, L, "replay", "state",
                                             &state_diff);
            }
            ref2_compared = state_read;
        }
        fprintf(stderr,
                "ds4: dflash verify compare pos=%u rows=%u "
                "bf16_mm=%d indexer_tiled=%d ref1_ms=%.3f candidate_ms=%.3f "
                "ref2_ms=%.3f forward=%d/%d/%d capture=0x%x/0x%x/0x%x\n",
                pos, L, verify_bf16_mm_requested,
                verify_indexer_tiled_requested,
                ref1_ms, candidate_ms, ref2_ms,
                ref1_forward, candidate_forward, ref2_forward,
                ref_capture_mask, candidate_capture_mask, ref2_capture_mask);
        verified = ref1_saved && candidate_saved && candidate_compared &&
            restored && ref2_saved && ref2_compared;
        ds4_gpu_tensor_free(verify_ref_state);
        ds4_gpu_tensor_free(verify_other_state);
        free(verify_ref_hc);
        free(verify_ref_capture);
        free(verify_other_capture);
    } else
#endif
    {
        verified = dflash_target_verify_forward(
            e, g, toks, pos, L, hc_rows_host,
            verify_candidate && !verify_compare_requested &&
                verify_bf16_mm_requested,
            verify_candidate && !verify_compare_requested &&
                verify_indexer_tiled_requested);
    }
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (verify_ledger &&
        !ds4_gpu_kernel_ledger_window_dump(verify_ledger_path)) {
        fprintf(stderr, "ds4: dflash verifier ledger dump failed: %s\n",
                verify_ledger_path);
    }
#endif
    const bool stepsnap_complete =
        ds4_gpu_glm53_kda_stepsnap_complete(kda_bytes) &&
        ds4_gpu_glm53_tail_stepsnap_complete();
    dflash_cycle_disarm();
    if (verified && !stepsnap_complete && dflash_stats_detail()) {
        fprintf(stderr, "ds4: dflash incomplete prefix capture; using replay\n");
    }
    const uint64_t t_verify = dflash_clock_now_ns();
    trace->verify_ms = dflash_clock_ms(t_draft, t_verify);
    if (!verified || dflash_fail_injected(DS4_DFLASH_FAIL_AFTER_VERIFY)) {
        dflash_cycle_fail_after_verify(s, g, state_saved);
        snprintf(err, errlen, "dflash: verify forward failed");
        return -1;
    }

    /* Per-row reference arithmetic, with NT4 sharing weights across rows.
     * HC collapse and normalization are common to both head paths. */
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
    /* NT4 defaults on only for the measured device/model profile. A force
     * broadens that profile, never the kernel shape; the kill dominates both
     * dispatch and diagnostics. Compare/bench retain scalar acceptance. */
    /* Partial-width verification pads the vocabulary head to the measured
     * eight-row NT4 shape by default; DS4_DFLASH_HEAD_PADDED=0 keeps the
     * scalar per-row head for widths below eight. */
    const bool head_pad = L >= 2u && L < 8u &&
        dflash_verify_default_on("DS4_DFLASH_HEAD_PADDED");
    const uint32_t head_rows = head_pad ? 8u : L;
    const bool head_shape = H == 4096u && DS4_N_VOCAB == 154880u && head_rows == 8u &&
        e->weights.output->type == DS4_TENSOR_Q8_0;
    const bool head_killed = getenv("DS4_DFLASH_DISABLE_HEAD_NT4") != NULL;
    const bool head_compare = head_shape && !head_killed &&
        (getenv("DS4_DFLASH_HEAD_COMPARE") || getenv("DS4_DFLASH_HEAD_BENCH"));
    const bool head_use = !head_compare && dflash_fast_path_enabled(head_shape,
        dflash2_fast_default(&e->dflash_model, &e->dflash_weights,
                            &e->model, &e->weights),
        getenv("DS4_DFLASH_HEAD_NT4"),
        getenv("DS4_DFLASH_DISABLE_HEAD_NT4"), NULL);
    bool head_nt4_ran = false;
    if (head_compare && !g->dflash_head_compare) {
        g->dflash_head_compare = ds4_gpu_tensor_alloc(
            (uint64_t)8u * DS4_N_VOCAB * sizeof(float));
    }
    /* All heads in one command stream, one scalar-logit readback at the end. */
    {
        bool ok = g_dflash_pool.inited &&
                  L <= g_dflash_pool.spec_cap &&
                  (!head_compare || g->dflash_head_compare) &&
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
        /* Only the vocabulary head is padded. Extra normalized rows copy a
         * real finite row and are ignored; target verification, snapshots and
         * consumed token counts retain the actual L rows. */
        if (ok && head_pad && (head_use || head_compare)) {
            for (uint32_t r = L; ok && r < head_rows; r++)
                ok = ds4_gpu_tensor_copy(g_dflash_pool.batch_norm,
                    (uint64_t)r * H * sizeof(float), g_dflash_pool.batch_norm,
                    (uint64_t)(L - 1u) * H * sizeof(float),
                    (uint64_t)H * sizeof(float)) != 0;
        }
        if (ok && head_use) {
            head_nt4_ran = ds4_gpu_dflash_head_nt4_tensor(
                g_dflash_pool.batch_logits, e->model.map, e->model.size,
                e->weights.output->abs_offset, H, DS4_N_VOCAB,
                g_dflash_pool.batch_norm, head_rows) != 0;
        }
        if (ok && !head_nt4_ran) {
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
        if (ok && head_compare) {
            head_nt4_ran = ds4_gpu_dflash_head_nt4_tensor(
                g->dflash_head_compare, e->model.map, e->model.size,
                e->weights.output->abs_offset, H, DS4_N_VOCAB,
                g_dflash_pool.batch_norm, head_rows) != 0;
            ok = head_nt4_ran;
        }
        if (!ds4_gpu_end_commands()) ok = false;
        if (ok) {
            ok = ds4_gpu_tensor_read(
                    g_dflash_pool.batch_logits, 0, row_logits,
                    (uint64_t)L * DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (ok && head_compare) {
            static float *compare_logits;
            if (!compare_logits) compare_logits = xmalloc(
                (size_t)8u * DS4_N_VOCAB * sizeof(float));
            ok = ds4_gpu_tensor_read(g->dflash_head_compare, 0, compare_logits,
                (uint64_t)L * DS4_N_VOCAB * sizeof(float)) != 0;
            if (ok) dflash_head_compare_logits(pos, L, row_logits, compare_logits);
            if (ok) ok = dflash_head_bench(s, pos);
        }
        if (!ok) {
            dflash_cycle_fail_after_verify(s, g, state_saved);
            snprintf(err, errlen, "dflash: verify head failed");
            return -1;
        }
    }
    if (dflash_stats_detail()) {
        fprintf(stderr, "ds4: dflash head pos=%u rows=%u nt4=%d scalar_logits_retained=%d\n",
                pos, L, head_nt4_ran, !head_use || !head_nt4_ran);
    }

    const uint64_t t_heads = dflash_clock_now_ns();
    trace->heads_ms = dflash_clock_ms(t_verify, t_heads);
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
            /* The ignore-EOS API selects its first token with all generation
             * stop/control ids excluded. Apply the same contract to its
             * speculative suffix: leave this row as the next frontier so the
             * caller can select the best permitted token from its logits. */
            if (ignore_eos && ds4_token_is_stop_for_think_mode(
                    e, drafts[i], think_mode)) break;
            const int pred = glm_session_logits_argmax(
                    row_logits + (size_t)i * DS4_N_VOCAB);
            if (pred != drafts[i]) break;
            n_accept++;
        }
    }
    const uint32_t n_committed = (uint32_t)n_accept + 1u;
    trace->accepted = (uint32_t)n_accept;
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
        ds4_session_dflash_history_stage(s, n_committed, pos + n_committed);
    } else {
        s->dflash_pending_rows = 0;
        s->dflash_history.len = 0;
    }
    s->dflash_warm = 1;
    trace->tail_ms = dflash_clock_elapsed_ms(t_heads);
    if (dflash_stats_detail()) {
        fprintf(stderr,
                "ds4: dflash cycle drafted=%d accepted=%d rollback=%s "
                "ctx=%u draft=%.1fms verify=%.1fms heads=%.1fms tail=%.1fms "
                "serial=%.1fms cyc=%.1fms be=%.2f\n",
                n_draft, n_accept,
                plan.kind == DS4_DFLASH_ROLLBACK_NONE ? "none" :
                plan.kind == DS4_DFLASH_ROLLBACK_STEPSNAP ? "stepsnap" :
                plan.kind == DS4_DFLASH_ROLLBACK_REPLAY ? "replay" : "refuse",
                s->dflash_cache.n_layer ? s->dflash_cache.layer[0].len : 0u,
                dflash_clock_ms(t_cycle0, t_draft),
                dflash_clock_ms(t_draft, t_verify),
                dflash_clock_ms(t_verify, t_heads),
                dflash_clock_elapsed_ms(t_heads),
                s->dflash_serial_ms, s->dflash_cycle_ms, breakeven);
    }
    {
        const float cyc_ms = (float)(dflash_clock_elapsed_ms(cyc_t0));
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
        bool ignore_eos, ds4_think_mode think_mode,
        int *accepted, int accepted_cap,
        char *err, size_t errlen) {
    memset(&s->dflash_trace, 0, sizeof(s->dflash_trace));
    s->dflash_trace.confidence_first = s->dflash_trace.confidence_min = -1.0f;
    const uint64_t start = dflash_clock_now_ns();
    const uint32_t pos = (uint32_t)s->checkpoint.len;
    ds4_dflash_adaptive *a = &s->dflash_adaptive;
    ds4_dflash_adaptive_trace *trace = &s->dflash_trace;
    dflash_adaptive_before_call(a);
    trace->serial_consumed_before = a->serial_consumed;
    int rc;
    /* Full-block mode uses the same real consumed-prefix ledger, but its
     * measurements never gate drafting. Fault and capacity checks stay in
     * the cycle, independent of either public policy. */
    if (s->dflash_full_block || s->dflash_fault.disabled ||
            (a->active && !a->invalid && accepted_cap > 1)) {
        rc = ds4_session_glm_dflash_cycle_impl(s, first_token,
            0.0f, 0, 0.0f, 0.0f, NULL, ignore_eos, think_mode,
            accepted, accepted_cap, err, errlen);
    } else {
        trace->operation = "serial";
        s->dflash_pending_rows = 0;
        const uint64_t serial_start = dflash_clock_now_ns();
        rc = ds4_session_eval_internal(s, first_token, false, err, errlen);
        trace->serial_ms = dflash_clock_elapsed_ms(serial_start);
        if (rc == 0) { accepted[0] = first_token; rc = 1; }
        else rc = -1;
    }
    uint64_t actual_ns = 0;
    const uint64_t account_start = dflash_clock_now_ns();
    (void)dflash_clock_account_ns(start, account_start, s->dflash_clock_ns, &actual_ns);
    if (a->active) dflash_adaptive_finish(a, actual_ns,
            rc > 0 ? (uint32_t)rc : 0u, trace);
    if (dflash_stats_detail() && a->active) {
        fprintf(stderr, "ds4: dflash adaptive pos=%u operation=%s chosen=%u "
                "proposed=%u verified=%u accepted=%u returned=%d next=%u "
                "p_min=%.6f confidence_first=%.6f confidence_min=%.6f "
                "bootstrap_rows=%u setup_ms=%.6f draft_ms=%.6f verify_ms=%.6f "
                "heads_ms=%.6f tail_ms=%.6f serial_ms=%.6f total_ms=%.6f status=%s "
                "candidate=%u skipped=%d economic=%d calibration=%d serial_calibration=%d "
                "retry_remaining=%u limit=%u probe_only=%d recovery=%d "
                "funded=%d funding_bank_ms=%.6f funding_cost_ms=%.6f entry=%d serial_consumed_before=%llu\n",
                pos, trace->operation ? trace->operation : "serial", trace->chosen,
                trace->proposed, trace->verified, trace->accepted, rc > 0 ? rc : 0,
                dflash_adaptive_limit(a), a->config.p_min, trace->confidence_first,
                trace->confidence_min, trace->bootstrap_rows, trace->setup_ms,
                trace->draft_ms, trace->verify_ms, trace->heads_ms, trace->tail_ms,
                trace->serial_ms, (double)actual_ns / 1e6, rc > 0 ? "ok" : "error",
                trace->candidate, trace->skipped, trace->economic, trace->calibration,
                trace->serial_calibration, a->skip_remaining, dflash_adaptive_limit(a), a->probe_only, trace->recovery, trace->funded,
                trace->funding_bank_ms, trace->funding_cost_ms, trace->entry,
                (unsigned long long)trace->serial_consumed_before);
    }
    if (a->active) {
        uint64_t ns = 0;
        (void)dflash_clock_account_ns(account_start, dflash_clock_now_ns(), s->dflash_clock_ns, &ns);
        dflash_adaptive_charge(a, ns);
    }
    return rc;
}
#endif
