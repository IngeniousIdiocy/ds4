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
    uint64_t bound_gen;         /* generation the current prefill belongs to */
    uint64_t owner_gen;         /* generation whose rows the ring holds (0 = none) */
    uint32_t n_taps;
    uint32_t taps[DS4_DFLASH2_MAX_TARGET];
} ds4_glm_dflash_seed;
static ds4_glm_dflash_seed g_glm_dflash_seed;

/* Engine teardown, after the Metal drain; no capture owner may survive it. */
static void glm_dflash_seed_free(void) {
    ds4_gpu_tensor_free(g_glm_dflash_seed.buf);
    free(g_glm_dflash_seed.ring);
    memset(&g_glm_dflash_seed, 0, sizeof(g_glm_dflash_seed));
    memset(&g_glm_dflash_cap, 0, sizeof(g_glm_dflash_cap));
}

static void glm_dflash_seed_configure(const ds4_dflash2_weights *dw,
                                      bool speculative) {
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
    /* Capped requests cannot pay optional prefill capture from future decode
     * credit. They obtain fresh features only after funding a decode bundle. */
    g_glm_dflash_seed.enabled = speculative;
}

/* Stamp the prefill about to run with its session's conditioning generation,
 * so a ring filled for one request cannot be ingested by another. */
static void glm_dflash_seed_bind_owner(uint64_t gen) {
    g_glm_dflash_seed.bound_gen = gen;
}

/* Drop whatever the ring holds: a cancelled or failed prefill leaves a
 * partial capture that belongs to no completed prefix. */
static void glm_dflash_seed_invalidate(void) {
    g_glm_dflash_seed.ring_len = 0;
    g_glm_dflash_seed.ring_end_pos = 0;
    g_glm_dflash_seed.owner_gen = 0;
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
    if (!ok) {
        /* a failed or cancelled chunk leaves a partial capture that belongs
         * to no completed prefix */
        glm_dflash_seed_invalidate();
        return;
    }
    if (!sd->enabled) return;
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
    /* readback + ring upkeep: prefill-time cost, outside the adaptive
     * cycle timer, so account for it explicitly. */
    const double t0 = now_sec();
    if (ds4_gpu_tensor_read(sd->buf, (uint64_t)skip * row * sizeof(float),
                            sd->ring + (size_t)sd->ring_len * row,
                            (uint64_t)take * row * sizeof(float))) {
        sd->ring_len += take;
        sd->ring_end_pos = pos_end;
        sd->owner_gen = sd->bound_gen;
        g_dflash_overhead.seed_rows += take;
    } else {
        /* The in-place ring may already have dropped rows or received a
         * partial copy. Drop eligibility; the next cycle can refresh safely. */
        glm_dflash_seed_invalidate();
    }
    g_dflash_overhead.seed_readback_ms += (now_sec() - t0) * 1000.0;
    g_dflash_overhead.seed_chunks++;
}
#endif
