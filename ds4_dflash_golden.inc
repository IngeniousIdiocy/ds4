/* DFlash2 golden self-test: validates the drafter forward and selector
 * against numpy reference vectors (validated to 4e-6 against z-lab's own
 * MLX implementation with the real weights). Run with
 * DS4_DFLASH_GOLDEN_DIR=/tmp/dflash-goldens at engine load; exits after. */

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
static float *dflash_golden_read(const char *dir, const char *name,
                                 size_t elems) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "golden: missing %s\n", path);
        return NULL;
    }
    float *buf = xmalloc(elems * sizeof(float));
    const size_t got = fread(buf, sizeof(float), elems, f);
    fclose(f);
    if (got != elems) {
        fprintf(stderr, "golden: short read %s (%zu/%zu)\n", path, got, elems);
        free(buf);
        return NULL;
    }
    return buf;
}

static void dflash_golden_cmp(const char *what, const float *got,
                              const float *want, size_t n) {
    double max_rel = 0.0, max_abs = 0.0, scale = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)got[i] - (double)want[i]);
        const double s = fabs((double)want[i]);
        if (s > scale) scale = s;
        if (d > max_abs) { max_abs = d; worst = i; }
        const double r = d / (s > 1e-3 ? s : 1e-3);
        if (r > max_rel) max_rel = r;
    }
    fprintf(stderr,
            "golden %-14s max_abs=%.4g (scale %.4g) max_rel=%.4g "
            "got[%zu]=%.5g want=%.5g\n",
            what, max_abs, scale, max_rel, worst, got[worst], want[worst]);
}

static void dflash2_golden_selftest(
        const ds4_model *dm,
        const ds4_dflash2_weights *dw,
        const ds4_model *tm,
        const ds4_weights *tw) {
    const char *dir = getenv("DS4_DFLASH_GOLDEN_DIR");
    if (!dir || !dir[0]) return;
    const uint32_t H = dw->n_embd;
    const uint32_t NT = dw->n_target;
    const uint32_t L = 8, NC = 3;
    float *features = dflash_golden_read(dir, "features.f32bin",
                                         (size_t)NC * NT * H);
    float *want_ctx = dflash_golden_read(dir, "h_ctx.f32bin", (size_t)NC * H);
    float *block = dflash_golden_read(dir, "block_embeds.f32bin",
                                      (size_t)L * H);
    float *want_layers = dflash_golden_read(dir, "layer_hidden.f32bin",
                                            (size_t)5 * L * H);
    float *want_final = dflash_golden_read(dir, "final_hidden.f32bin",
                                           (size_t)L * H);
    if (!features || !want_ctx || !block || !want_layers || !want_final) {
        exit(1);
    }

    /* 1. context projection: fc + enc norm */
    float *fc_out = xmalloc((size_t)NC * H * sizeof(float));
    float *h_ctx = xmalloc((size_t)NC * H * sizeof(float));
    dflash2_matvecs(fc_out, dm, dw->fc, features, NC);
    for (uint32_t t = 0; t < NC; t++) {
        rms_norm_weight(h_ctx + (size_t)t * H, fc_out + (size_t)t * H,
                        tensor_data(dm, dw->enc_output_norm), H, dw->rms_eps);
    }
    dflash_golden_cmp("h_ctx", h_ctx, want_ctx, (size_t)NC * H);

    /* 2. layer-by-layer forward, block at pos 103..110, ctx at 100..102 */
    float *h = xmalloc((size_t)L * H * sizeof(float));
    memcpy(h, block, (size_t)L * H * sizeof(float));
    extern void ds4_gpu_glm53_bf16_mv_max_set(uint32_t v);
    ds4_gpu_glm53_bf16_mv_max_set(1);
    for (uint32_t il = 0; il < dw->n_layer; il++) {
        h_live = 0;
        if (!ds4_gpu_begin_commands() ||
            !dflash2_layer_forward_gpu(h, h_ctx, NC, NULL, il, dm,
                                       &dw->layer[il], dw, L, 103u) ||
            !dflash2_gpu_finish_h(h, L, H)) {
            fprintf(stderr, "golden: layer %u forward failed\n", il);
            exit(1);
        }
        char what[32];
        snprintf(what, sizeof(what), "layer%u", il);
        dflash_golden_cmp(what, h, want_layers + (size_t)il * L * H,
                          (size_t)L * H);
    }
    ds4_gpu_glm53_bf16_mv_max_set(8);

    /* 3. final norm */
    float *fin = xmalloc((size_t)L * H * sizeof(float));
    for (uint32_t t = 0; t < L; t++) {
        rms_norm_weight(fin + (size_t)t * H, h + (size_t)t * H,
                        tensor_data(dm, dw->output_norm), H, dw->rms_eps);
    }
    dflash_golden_cmp("final", fin, want_final, (size_t)L * H);

    /* 4. selector on fake logits */
    float *fake_logits = dflash_golden_read(dir, "fake_logits.f32bin",
                                            (size_t)7 * DS4_N_VOCAB);
    if (fake_logits && dflash2_selector_ready(dw, dm)) {
        int sel[7];
        /* reference used final-normed hiddens of block rows 1..7 */
        dflash2_selector_chain(dw, dm, fin + H, fake_logits, 42, sel, 7);
        fprintf(stderr, "golden selector tokens:");
        for (int i = 0; i < 7; i++) fprintf(stderr, " %d", sel[i]);
        fprintf(stderr, "  (want 85355 276 51201 69525 26492 27111 29933)\n");
    }
    (void)tm; (void)tw;
    fprintf(stderr, "golden: self-test complete\n");
    exit(0);
}
#endif
