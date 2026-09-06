/* DFlash prefill feature seeding: while enabled, every GLM prefill chunk's
 * tap-layer features are captured and appended to a CPU ring holding the
 * last (sliding_window - 1) positions. The first DFlash cycle ingests the
 * ring into the drafter's context-KV cache so drafting starts with the
 * prompt in view instead of blind. Single-session (CLI/eval) scope. */

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
typedef struct {
    int enabled;
    ds4_gpu_tensor *buf;        /* [chunk_cap][n_taps][embd] capture target */
    uint32_t chunk_cap;
    float *ring;                /* [ring_cap][n_taps][embd] newest-last */
    uint32_t ring_len;
    uint32_t ring_cap;
    uint32_t ring_end_pos;      /* absolute position AFTER the last ring row */
    uint32_t n_taps;
    uint32_t taps[DS4_DFLASH2_MAX_TARGET];
} ds4_glm_dflash_seed;
static ds4_glm_dflash_seed g_glm_dflash_seed;

static void glm_dflash_seed_configure(const ds4_dflash2_weights *dw) {
    if (!dw || !dw->ready) return;
    g_glm_dflash_seed.n_taps = dw->n_target;
    for (uint32_t j = 0; j < dw->n_target; j++) {
        g_glm_dflash_seed.taps[j] = dw->target_layers[j];
    }
    g_glm_dflash_seed.ring_cap =
        dw->sliding_window ? dw->sliding_window - 1u : 2047u;
    if (g_glm_dflash_seed.ring_cap > 256u) g_glm_dflash_seed.ring_cap = 256u;
    {
        const char *cenv = getenv("DS4_DFLASH_CTX_CAP");
        if (cenv && atoi(cenv) > 0) {
            g_glm_dflash_seed.ring_cap = (uint32_t)atoi(cenv);
        }
    }
    g_glm_dflash_seed.chunk_cap = glm53_prefill_chunk_tokens();
    g_glm_dflash_seed.enabled = 1;
}

static bool glm_dflash_seed_chunk_begin(uint32_t chunk) {
    ds4_glm_dflash_seed *sd = &g_glm_dflash_seed;
    /* Arm only for real prefill chunks. Decode tokens (chunk 1) and verify
     * blocks (chunk <= 8) must not pay the capture + readback cost; the
     * cycle feeds the drafter its own captures during decode. Keeping this
     * armed across requests lets every new prefill refill the ring, so a
     * rewound session (server slot reuse) re-warms from the fresh prompt. */
    if (!sd->enabled || chunk < 32u || chunk > sd->chunk_cap) return false;
    if (!sd->buf) {
        sd->buf = ds4_gpu_tensor_alloc((uint64_t)sd->chunk_cap * sd->n_taps *
                                       DS4_N_EMBD * sizeof(float));
    }
    if (!sd->ring) {
        sd->ring = xmalloc((size_t)sd->ring_cap * sd->n_taps * DS4_N_EMBD *
                           sizeof(float));
        sd->ring_len = 0;
    }
    if (!sd->buf || !sd->ring) return false;
    g_glm_dflash_cap.enabled = 1;
    g_glm_dflash_cap.buf = sd->buf;
    g_glm_dflash_cap.n_rows = chunk;
    g_glm_dflash_cap.n_taps = sd->n_taps;
    for (uint32_t j = 0; j < sd->n_taps; j++) {
        g_glm_dflash_cap.taps[j] = sd->taps[j];
    }
    return true;
}

static void glm_dflash_seed_chunk_end(uint32_t chunk, uint32_t pos_end,
                                      bool ok) {
    ds4_glm_dflash_seed *sd = &g_glm_dflash_seed;
    g_glm_dflash_cap.enabled = 0;
    if (!ok || !sd->enabled) return;
    const uint64_t row = (uint64_t)sd->n_taps * DS4_N_EMBD;
    uint32_t take = chunk;
    uint32_t skip = 0;
    if (take > sd->ring_cap) {
        skip = take - sd->ring_cap;
        take = sd->ring_cap;
    }
    if (sd->ring_len + take > sd->ring_cap) {
        const uint32_t drop = sd->ring_len + take - sd->ring_cap;
        memmove(sd->ring, sd->ring + (size_t)drop * row,
                (size_t)(sd->ring_len - drop) * row * sizeof(float));
        sd->ring_len -= drop;
    }
    if (ds4_gpu_tensor_read(sd->buf, (uint64_t)skip * row * sizeof(float),
                            sd->ring + (size_t)sd->ring_len * row,
                            (uint64_t)take * row * sizeof(float))) {
        sd->ring_len += take;
        sd->ring_end_pos = pos_end;
    }
}
#endif
