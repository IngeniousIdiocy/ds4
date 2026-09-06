/*
 * Direct synthetic coverage for the KDA paths that the upstream-recipe
 * (all-Q8_0) weight layout newly reaches.  No model file, no prefill; it opens
 * a Metal device and dispatches the real kernels against a hand-built model
 * map at the production GLM-5.3 shape (head dim 128, low-rank rank 128).
 *
 * What is compared, and how strictly:
 *
 *   A. out-tail fold only (do_prologue=0, do_out=1) vs the two-dispatch
 *      reference ds4_gpu_glm53_kda_decode_split2, which the sibling harness
 *      tests/test_glm53_kda.c already certifies bit-identical to the fused
 *      ds4_gpu_glm53_kda_decode.  EXACT-ORDER: compared word for word, on the
 *      output AND on both pieces of recurrent state (the conv ring and the
 *      delta-rule state matrix) after every step.
 *
 *   B. the Q8_0 six-way fold ds4_gpu_matmul_q8_0_kda6_flat_tensor vs six
 *      standalone ds4_gpu_matmul_q8_0_tensor dispatches.  EXACT-ORDER.
 *      (ds4_gpu_matmul_q8_0_tensor is the same entry the decode ladder reaches
 *      through glm_graph_matmul_q8_0_decode_tensor ->
 *      ds4_gpu_matmul_quant_decode_mpp_model_view_tensor ->
 *      ds4_gpu_matmul_quant_impl_tensor -> ds4_gpu_matmul_q8_0_legacy_tensor,
 *      so this is the real standalone arm, not a lookalike.)
 *
 *   C. the Q8_0 f_b/g_b pair fusion ds4_gpu_matmul_q8_0_pair2in_tensor vs two
 *      standalone dispatches.  EXACT-ORDER.
 *
 *   D. Q8_0 glue prologue + out fold (do_prologue=1, do_out=1, lr_q8=1) vs
 *      pair2in + split2.  EXACT-ORDER on the two gate buffers the prologue
 *      produces, on the output, and on both pieces of recurrent state.
 *
 *   E. refusals: the callee must return 0 AND encode nothing (poisoned outputs
 *      stay poisoned) for a Q8_0 rank whose block count exceeds NQ (nb > 8),
 *      for a rank that is not a multiple of 32, for do_prologue==0 &&
 *      do_out==0, and for a fold extent that is not a multiple of nr0.
 *
 * Every comparison in this file is exact-order and therefore bit-identical
 * (memcmp), including NaN payloads: the whole point of these paths is that
 * they only re-partition work across threadgroups without changing any
 * summation order.  Nothing here is compared with a tolerance.
 *
 * Operands are run twice: a "tame" set (well-conditioned, additionally
 * required to stay finite) and an "adversarial" set (half scales at the
 * denormal 2^-24 and at 65504, int8 quants pinned at -128/0/127, activations
 * mixing 1e18f, denormal 1e-40f, -0.0f and alternating signs).  The
 * adversarial set is only required to be bit-identical; it may legitimately
 * produce inf/nan, and identical inf/nan is exactly what a re-partitioning
 * must preserve.
 *
 * Multi-head addressing is exercised with 4 heads, so every head but the first
 * has a non-zero base offset into the projection, the gates, the conv ring and
 * the state matrix.  Five decode steps run in sequence so the conv ring wraps
 * its 3-slot history and the state matrix carries.
 *
 * Build: make tests/test_glm53_kda_q8     Run: make test-glm53-kda-q8
 * It needs a Metal device (so: under the GPU lock) but no model weights.
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static int g_failures = 0;

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FATAL: %s failed\n", what);
        exit(1);
    }
}

/* Exact-order comparison: word for word, NaN payloads included. */
static void require_identical(const char *what, const void *a, const void *b,
                              size_t bytes) {
    if (memcmp(a, b, bytes) == 0) return;
    const uint32_t *x = (const uint32_t *)a;
    const uint32_t *y = (const uint32_t *)b;
    size_t first = 0;
    while (first < bytes / 4u && x[first] == y[first]) first++;
    fprintf(stderr,
            "MISMATCH: %s: first differing word %zu of %zu: 0x%08x vs 0x%08x\n",
            what, first, bytes / 4u, x[first], y[first]);
    g_failures++;
}

static void require_all_finite(const char *what, const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(v[i])) {
            fprintf(stderr, "NON-FINITE: %s: element %zu = %g\n",
                    what, i, v[i]);
            g_failures++;
            return;
        }
    }
}

/* ---- shape: the production GLM-5.3 KDA geometry, four heads ---- */
enum {
    D           = 128,              /* head dim, fixed by the kernels        */
    HEADS       = 4,
    PROJECTION  = HEADS * D,        /* 512                                   */
    RANK        = 128,              /* f_b / g_b input rank -> nb = 4        */
    BAD_RANK    = 288,              /* nb = 9 > NQ: must be refused          */
    EMBD        = 256,              /* q/k/v/f_a/beta/g_a input dim          */
    TOKENS      = 5,
    ROW272      = (EMBD / 32) * 34, /* Q8_0 row bytes at EMBD                */
    ROW136      = (RANK / 32) * 34, /* Q8_0 row bytes at RANK                */
    ROW306      = (BAD_RANK / 32) * 34,

    O_Q_CONV    = 0,
    O_K_CONV    = 8192,
    O_V_CONV    = 16384,
    O_A_LOG     = 24576,
    O_DT_BIAS   = 24832,
    O_NORM      = 26880,
    O_W_Q       = 27392,
    O_W_K       = 166656,
    O_W_V       = 305920,
    O_W_FA      = 445184,
    O_W_BETA    = 480000,
    O_W_GA      = 481280,
    O_W_FB      = 516096,
    O_W_GB      = 585728,
    O_W_FB9     = 655360,
    O_W_GB9     = 812288,
    MODEL_BYTES = 1048576,

    /* every output tensor is over-allocated by this many floats, filled with
     * a poison pattern, and checked afterwards */
    GUARD       = 64,
    POISON      = (int)0xDEADBEEF,
};

/* ---- Q8_0 block, as the kernels see it ---- */
typedef struct {
    uint16_t d;         /* half */
    int8_t   qs[32];
} test_block_q8_0;

/* Half bit patterns chosen so no float conversion is needed and the
 * adversarial set really does reach the edges of the format. */
static const uint16_t k_tame_d[] = { 0x3c00u, 0x3800u, 0xbc00u, 0x3400u };
static const uint16_t k_adv_d[]  = {
    0x0001u,  /* 2^-24, smallest denormal half */
    0x7bffu,  /* 65504, largest finite half    */
    0xfbffu,  /* -65504                        */
    0x0400u,  /* 2^-14, smallest normal half   */
    0x3c00u,  /* 1                             */
    0x8000u,  /* -0.0                          */
};

static void fill_q8(uint8_t *base, uint32_t rows, uint32_t in_dim,
                    uint32_t seed, int adversarial) {
    test_block_q8_0 *blocks = (test_block_q8_0 *)base;
    const uint32_t nb = in_dim / 32u;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t b = 0; b < nb; b++) {
            test_block_q8_0 *blk = blocks + (size_t)r * nb + b;
            const uint32_t mix = seed + r * 7919u + b * 104729u;
            if (adversarial) {
                blk->d = k_adv_d[mix % (sizeof(k_adv_d) / sizeof(k_adv_d[0]))];
                for (uint32_t i = 0; i < 32u; i++) {
                    const uint32_t s = (mix + i * 31u) % 5u;
                    blk->qs[i] = (int8_t)(s == 0 ? -128 :
                                          s == 1 ?  127 :
                                          s == 2 ?    0 :
                                          s == 3 ?   -1 : 1);
                }
            } else {
                blk->d = k_tame_d[mix % (sizeof(k_tame_d) / sizeof(k_tame_d[0]))];
                for (uint32_t i = 0; i < 32u; i++) {
                    blk->qs[i] = (int8_t)((int)((mix + i * 13u) % 21u) - 10);
                }
            }
        }
    }
}

static void fill_activations(float *dst, uint32_t n, uint32_t seed,
                             int adversarial) {
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t mix = seed + i * 2654435761u;
        if (adversarial) {
            switch (mix % 7u) {
                case 0: dst[i] =  1.0e18f;  break;
                case 1: dst[i] = -1.0e18f;  break;
                case 2: dst[i] =  1.0e-40f; break; /* denormal float */
                case 3: dst[i] = -0.0f;     break;
                case 4: dst[i] =  0.0f;     break;
                case 5: dst[i] = -3.7e-3f;  break;
                default: dst[i] = 2.9e-2f;  break;
            }
        } else {
            dst[i] = 0.05f * (float)((int)(mix % 41u) - 20) +
                     ((mix & 1u) ? 0.011f : -0.013f);
        }
    }
}

/* ---- guarded tensors ---- */
typedef struct {
    ds4_gpu_tensor *t;
    uint64_t        live_floats;   /* what the kernel may write */
    uint64_t        total_floats;
} guarded;

static guarded guarded_alloc(uint64_t live_floats) {
    guarded g;
    g.live_floats = live_floats;
    g.total_floats = live_floats + GUARD;
    g.t = ds4_gpu_tensor_alloc(g.total_floats * sizeof(float));
    require_ok(g.t != NULL, "guarded tensor allocation");
    return g;
}

static void guarded_poison(guarded *g) {
    static int32_t pattern[4096];
    for (size_t i = 0; i < sizeof(pattern) / sizeof(pattern[0]); i++) {
        pattern[i] = POISON;
    }
    uint64_t done = 0;
    while (done < g->total_floats) {
        const uint64_t chunk =
            (g->total_floats - done) < 4096u ? (g->total_floats - done) : 4096u;
        require_ok(ds4_gpu_tensor_write(g->t, done * sizeof(float), pattern,
                                        chunk * sizeof(float)),
                   "poison write");
        done += chunk;
    }
}

static void guarded_check_tail(const guarded *g, const char *what) {
    int32_t tail[GUARD];
    require_ok(ds4_gpu_tensor_read(g->t, g->live_floats * sizeof(float),
                                   tail, sizeof(tail)),
               "guard tail read");
    for (int i = 0; i < GUARD; i++) {
        if (tail[i] != (int32_t)POISON) {
            fprintf(stderr,
                    "OUT-OF-BOUNDS WRITE: %s: guard word %d = 0x%08x\n",
                    what, i, (uint32_t)tail[i]);
            g_failures++;
            return;
        }
    }
}

static void guarded_read(const guarded *g, float *dst) {
    require_ok(ds4_gpu_tensor_read(g->t, 0, dst,
                                   g->live_floats * sizeof(float)),
               "guarded read");
}

/* ---- shared state ---- */
static uint8_t *g_model;

static void reset_state(ds4_gpu_tensor *conv, ds4_gpu_tensor *state) {
    require_ok(ds4_gpu_tensor_fill_f32(conv, 0.0f,
                                       9u * (uint64_t)PROJECTION),
               "conv reset");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f,
                                       (uint64_t)HEADS * D * D),
               "state reset");
}

/* The weight regions are laid out by hand; make sure they neither overlap nor
 * run off the end of the map before anything is written into them. */
static void check_layout(void) {
    const struct { const char *name; uint64_t off, bytes; } r[] = {
        { "q_conv",  O_Q_CONV,  (uint64_t)PROJECTION * 4u * sizeof(float) },
        { "k_conv",  O_K_CONV,  (uint64_t)PROJECTION * 4u * sizeof(float) },
        { "v_conv",  O_V_CONV,  (uint64_t)PROJECTION * 4u * sizeof(float) },
        { "a_log",   O_A_LOG,   (uint64_t)HEADS * sizeof(float) },
        { "dt_bias", O_DT_BIAS, (uint64_t)PROJECTION * sizeof(float) },
        { "o_norm",  O_NORM,    (uint64_t)D * sizeof(float) },
        { "w_q",     O_W_Q,     (uint64_t)PROJECTION * ROW272 },
        { "w_k",     O_W_K,     (uint64_t)PROJECTION * ROW272 },
        { "w_v",     O_W_V,     (uint64_t)PROJECTION * ROW272 },
        { "w_f_a",   O_W_FA,    (uint64_t)RANK * ROW272 },
        { "w_beta",  O_W_BETA,  (uint64_t)HEADS * ROW272 },
        { "w_g_a",   O_W_GA,    (uint64_t)RANK * ROW272 },
        { "w_f_b",   O_W_FB,    (uint64_t)PROJECTION * ROW136 },
        { "w_g_b",   O_W_GB,    (uint64_t)PROJECTION * ROW136 },
        { "w_f_b9",  O_W_FB9,   (uint64_t)PROJECTION * ROW306 },
        { "w_g_b9",  O_W_GB9,   (uint64_t)PROJECTION * ROW306 },
    };
    const size_t n = sizeof(r) / sizeof(r[0]);
    for (size_t i = 0; i < n; i++) {
        if (r[i].off + r[i].bytes > (uint64_t)MODEL_BYTES) {
            fprintf(stderr, "layout: %s runs past the map\n", r[i].name);
            exit(1);
        }
        for (size_t j = i + 1; j < n; j++) {
            if (r[i].off < r[j].off + r[j].bytes &&
                r[j].off < r[i].off + r[i].bytes) {
                fprintf(stderr, "layout: %s overlaps %s\n",
                        r[i].name, r[j].name);
                exit(1);
            }
        }
    }
}

int main(void) {
    check_layout();
    g_model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    if (g_model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(g_model, MODEL_BYTES),
               "model map registration");

    /* activation-side buffers */
    guarded gq  = guarded_alloc(PROJECTION);
    guarded gk  = guarded_alloc(PROJECTION);
    guarded gv  = guarded_alloc(PROJECTION);
    guarded gx  = guarded_alloc(EMBD);          /* attn_norm row  */
    guarded glf = guarded_alloc(RANK);          /* f_a output     */
    guarded glg = guarded_alloc(RANK);          /* g_a output     */
    /* oversized low-rank rows, so the refusal cases below are refused by the
     * check under test and not by a too-small buffer */
    guarded gbig_f = guarded_alloc(BAD_RANK);
    guarded gbig_g = guarded_alloc(BAD_RANK);
    guarded gbeta_ref = guarded_alloc(HEADS);
    guarded gbeta_fus = guarded_alloc(HEADS);
    guarded gq_fus = guarded_alloc(PROJECTION);
    guarded gk_fus = guarded_alloc(PROJECTION);
    guarded gv_fus = guarded_alloc(PROJECTION);
    guarded glf_fus = guarded_alloc(RANK);
    guarded glg_fus = guarded_alloc(RANK);
    guarded ggate_ref = guarded_alloc(PROJECTION);
    guarded gogate_ref = guarded_alloc(PROJECTION);
    guarded ggate_cand = guarded_alloc(PROJECTION);
    guarded gogate_cand = guarded_alloc(PROJECTION);
    guarded gout_ref = guarded_alloc(PROJECTION);
    guarded gout_cand = guarded_alloc(PROJECTION);

    ds4_gpu_tensor *conv_ref = ds4_gpu_tensor_alloc(
        9u * (uint64_t)PROJECTION * sizeof(float));
    ds4_gpu_tensor *conv_cand = ds4_gpu_tensor_alloc(
        9u * (uint64_t)PROJECTION * sizeof(float));
    ds4_gpu_tensor *state_ref = ds4_gpu_tensor_alloc(
        (uint64_t)HEADS * D * D * sizeof(float));
    ds4_gpu_tensor *state_cand = ds4_gpu_tensor_alloc(
        (uint64_t)HEADS * D * D * sizeof(float));
    ds4_gpu_tensor *scratch_ref = ds4_gpu_tensor_alloc(
        (uint64_t)HEADS * (516u + D) * sizeof(float));
    ds4_gpu_tensor *scratch_cand = ds4_gpu_tensor_alloc(
        (uint64_t)HEADS * (516u + D) * sizeof(float));
    require_ok(conv_ref && conv_cand && state_ref && state_cand &&
               scratch_ref && scratch_cand, "state allocation");

    static float host_a[9u * PROJECTION];
    static float host_b[9u * PROJECTION];
    static float state_a[(size_t)HEADS * D * D];
    static float state_b[(size_t)HEADS * D * D];
    static float host_x[EMBD];
    static float host_p[PROJECTION];

    for (int adversarial = 0; adversarial <= 1; adversarial++) {
        const char *set = adversarial ? "adversarial" : "tame";

        /* ---- weights ---- */
        float *q_conv = (float *)(g_model + O_Q_CONV);
        float *k_conv = (float *)(g_model + O_K_CONV);
        float *v_conv = (float *)(g_model + O_V_CONV);
        float *a_log  = (float *)(g_model + O_A_LOG);
        float *dt     = (float *)(g_model + O_DT_BIAS);
        float *onorm  = (float *)(g_model + O_NORM);
        for (uint32_t c = 0; c < PROJECTION; c++) {
            for (uint32_t w = 0; w < 4u; w++) {
                q_conv[c * 4u + w] = 0.11f + 0.007f * (float)((c + w) % 9u);
                k_conv[c * 4u + w] = -0.09f + 0.005f * (float)((c + 2u * w) % 7u);
                v_conv[c * 4u + w] = 0.13f - 0.006f * (float)((c + 3u * w) % 11u);
            }
            dt[c] = -0.02f + 0.001f * (float)(c % 17u);
        }
        for (uint32_t h = 0; h < HEADS; h++) {
            a_log[h] = -0.3f + 0.2f * (float)h;
        }
        for (uint32_t i = 0; i < D; i++) {
            onorm[i] = 0.8f + 0.003f * (float)i;
        }
        fill_q8(g_model + O_W_Q,    PROJECTION, EMBD,     11u, adversarial);
        fill_q8(g_model + O_W_K,    PROJECTION, EMBD,     22u, adversarial);
        fill_q8(g_model + O_W_V,    PROJECTION, EMBD,     33u, adversarial);
        fill_q8(g_model + O_W_FA,   RANK,       EMBD,     44u, adversarial);
        fill_q8(g_model + O_W_BETA, HEADS,      EMBD,     55u, adversarial);
        fill_q8(g_model + O_W_GA,   RANK,       EMBD,     66u, adversarial);
        fill_q8(g_model + O_W_FB,   PROJECTION, RANK,     77u, adversarial);
        fill_q8(g_model + O_W_GB,   PROJECTION, RANK,     88u, adversarial);
        fill_q8(g_model + O_W_FB9,  PROJECTION, BAD_RANK, 99u, adversarial);
        fill_q8(g_model + O_W_GB9,  PROJECTION, BAD_RANK, 111u, adversarial);

        /* =============================================================
         * B. Q8_0 six-way fold vs six standalone dispatches
         * ============================================================= */
        fill_activations(host_x, EMBD, 1234u + (uint32_t)adversarial, adversarial);
        require_ok(ds4_gpu_tensor_write(gx.t, 0, host_x, sizeof(host_x)),
                   "attn_norm write");

        guarded *ref_outs[6] = { &gq, &gk, &gv, &glf, &gbeta_ref, &glg };
        guarded *fus_outs[6] = { &gq_fus, &gk_fus, &gv_fus, &glf_fus,
                                 &gbeta_fus, &glg_fus };
        const uint64_t offs[6] = { O_W_Q, O_W_K, O_W_V,
                                   O_W_FA, O_W_BETA, O_W_GA };
        const uint64_t dims[6] = { PROJECTION, PROJECTION, PROJECTION,
                                   RANK, HEADS, RANK };
        for (int i = 0; i < 6; i++) {
            guarded_poison(ref_outs[i]);
            guarded_poison(fus_outs[i]);
        }
        for (int i = 0; i < 6; i++) {
            require_ok(ds4_gpu_matmul_q8_0_tensor(ref_outs[i]->t, g_model,
                                                  MODEL_BYTES, offs[i], EMBD,
                                                  dims[i], gx.t, 1),
                       "standalone Q8_0 projection");
        }
        {
            ds4_gpu_tensor *fused[6];
            for (int i = 0; i < 6; i++) fused[i] = fus_outs[i]->t;
            require_ok(ds4_gpu_matmul_q8_0_kda6_flat_tensor(
                           fused, g_model, MODEL_BYTES, offs, EMBD, dims, gx.t),
                       "Q8_0 kda6 flat fold");
        }
        for (int i = 0; i < 6; i++) {
            static float a[PROJECTION], b[PROJECTION];
            guarded_read(ref_outs[i], a);
            guarded_read(fus_outs[i], b);
            char what[96];
            snprintf(what, sizeof(what),
                     "[%s] B: kda6_flat slice %d vs standalone", set, i);
            require_identical(what, a, b,
                              (size_t)ref_outs[i]->live_floats * sizeof(float));
            if (!adversarial) {
                require_all_finite(what, a, (size_t)ref_outs[i]->live_floats);
            }
            guarded_check_tail(ref_outs[i], what);
            guarded_check_tail(fus_outs[i], what);
        }

        /* =============================================================
         * C. Q8_0 pair2in vs two standalone dispatches
         * ============================================================= */
        guarded_poison(&ggate_ref);
        guarded_poison(&gogate_ref);
        guarded_poison(&ggate_cand);
        guarded_poison(&gogate_cand);
        require_ok(ds4_gpu_matmul_q8_0_tensor(ggate_ref.t, g_model, MODEL_BYTES,
                                              O_W_FB, RANK, PROJECTION,
                                              glf.t, 1),
                   "standalone Q8_0 f_b");
        require_ok(ds4_gpu_matmul_q8_0_tensor(gogate_ref.t, g_model, MODEL_BYTES,
                                              O_W_GB, RANK, PROJECTION,
                                              glg.t, 1),
                   "standalone Q8_0 g_b");
        require_ok(ds4_gpu_matmul_q8_0_pair2in_tensor(
                       ggate_cand.t, gogate_cand.t, g_model, MODEL_BYTES,
                       O_W_FB, O_W_GB, RANK, PROJECTION, glf.t, glg.t),
                   "Q8_0 pair2in fusion");
        {
            static float a[PROJECTION], b[PROJECTION];
            guarded_read(&ggate_ref, a);
            guarded_read(&ggate_cand, b);
            char what[96];
            snprintf(what, sizeof(what), "[%s] C: pair2in f_b vs standalone", set);
            require_identical(what, a, b, sizeof(a));
            guarded_read(&gogate_ref, a);
            guarded_read(&gogate_cand, b);
            snprintf(what, sizeof(what), "[%s] C: pair2in g_b vs standalone", set);
            require_identical(what, a, b, sizeof(a));
            guarded_check_tail(&ggate_cand, "C pair2in f_b");
            guarded_check_tail(&gogate_cand, "C pair2in g_b");
        }

        /* =============================================================
         * A. out-tail fold only vs split2, over TOKENS decode steps
         * ============================================================= */
        reset_state(conv_ref, state_ref);
        reset_state(conv_cand, state_cand);
        for (uint32_t t = 0; t < TOKENS; t++) {
            fill_activations(host_p, PROJECTION, 900u + t * 17u + (uint32_t)adversarial * 5u, adversarial);
            require_ok(ds4_gpu_tensor_write(gq.t, 0, host_p, sizeof(host_p)), "A q write");
            fill_activations(host_p, PROJECTION, 901u + t * 19u + (uint32_t)adversarial * 5u, adversarial);
            require_ok(ds4_gpu_tensor_write(gk.t, 0, host_p, sizeof(host_p)), "A k write");
            fill_activations(host_p, PROJECTION, 902u + t * 23u + (uint32_t)adversarial * 5u, adversarial);
            require_ok(ds4_gpu_tensor_write(gv.t, 0, host_p, sizeof(host_p)), "A v write");
            fill_activations(host_p, PROJECTION, 903u + t * 29u + (uint32_t)adversarial * 5u, adversarial);
            require_ok(ds4_gpu_tensor_write(ggate_ref.t, 0, host_p, sizeof(host_p)), "A gate write");
            fill_activations(host_p, PROJECTION, 904u + t * 31u + (uint32_t)adversarial * 5u, adversarial);
            require_ok(ds4_gpu_tensor_write(gogate_ref.t, 0, host_p, sizeof(host_p)), "A ogate write");
            fill_activations(host_p, HEADS, 905u + t * 37u + (uint32_t)adversarial * 5u, adversarial);
            require_ok(ds4_gpu_tensor_write(gbeta_ref.t, 0, host_p,
                                            HEADS * sizeof(float)), "A beta write");

            guarded_poison(&gout_ref);
            guarded_poison(&gout_cand);
            require_ok(ds4_gpu_glm53_kda_decode_split2(
                           gout_ref.t, conv_ref, state_ref, scratch_ref,
                           gq.t, gk.t, gv.t, ggate_ref.t, gbeta_ref.t,
                           gogate_ref.t, g_model, MODEL_BYTES,
                           O_Q_CONV, O_K_CONV, O_V_CONV, O_A_LOG, O_DT_BIAS,
                           O_NORM, HEADS, 1, -5.0f, 1e-5f),
                       "A reference split2");
            require_ok(ds4_gpu_glm53_kda_decode_glue(
                           gout_cand.t, conv_cand, state_cand, scratch_cand,
                           gq.t, gk.t, gv.t, ggate_ref.t, gbeta_ref.t,
                           gogate_ref.t, glf.t, glg.t, g_model, MODEL_BYTES,
                           O_Q_CONV, O_K_CONV, O_V_CONV, O_A_LOG, O_DT_BIAS,
                           O_NORM, O_W_FB, O_W_GB, RANK, /*lr_q8=*/0,
                           HEADS, 1, /*do_prologue=*/0, /*do_out=*/1,
                           -5.0f, 1e-5f),
                       "A candidate glue out-fold-only");

            char what[96];
            snprintf(what, sizeof(what), "[%s] A: out fold, token %u, output", set, t);
            guarded_read(&gout_ref, host_a);
            guarded_read(&gout_cand, host_b);
            require_identical(what, host_a, host_b, PROJECTION * sizeof(float));
            if (!adversarial) require_all_finite(what, host_a, PROJECTION);
            guarded_check_tail(&gout_cand, what);

            snprintf(what, sizeof(what), "[%s] A: out fold, token %u, conv ring", set, t);
            require_ok(ds4_gpu_tensor_read(conv_ref, 0, host_a, sizeof(host_a)), "conv ref read");
            require_ok(ds4_gpu_tensor_read(conv_cand, 0, host_b, sizeof(host_b)), "conv cand read");
            require_identical(what, host_a, host_b, sizeof(host_a));

            snprintf(what, sizeof(what), "[%s] A: out fold, token %u, delta-rule state", set, t);
            require_ok(ds4_gpu_tensor_read(state_ref, 0, state_a, sizeof(state_a)), "state ref read");
            require_ok(ds4_gpu_tensor_read(state_cand, 0, state_b, sizeof(state_b)), "state cand read");
            require_identical(what, state_a, state_b, sizeof(state_a));
        }

        /* =============================================================
         * D. Q8_0 prologue + out fold vs pair2in + split2, over TOKENS steps
         * ============================================================= */
        reset_state(conv_ref, state_ref);
        reset_state(conv_cand, state_cand);
        for (uint32_t t = 0; t < TOKENS; t++) {
            fill_activations(host_p, PROJECTION, 700u + t * 13u + (uint32_t)adversarial * 3u, adversarial);
            require_ok(ds4_gpu_tensor_write(gq.t, 0, host_p, sizeof(host_p)), "D q write");
            fill_activations(host_p, PROJECTION, 701u + t * 41u + (uint32_t)adversarial * 3u, adversarial);
            require_ok(ds4_gpu_tensor_write(gk.t, 0, host_p, sizeof(host_p)), "D k write");
            fill_activations(host_p, PROJECTION, 702u + t * 43u + (uint32_t)adversarial * 3u, adversarial);
            require_ok(ds4_gpu_tensor_write(gv.t, 0, host_p, sizeof(host_p)), "D v write");
            fill_activations(host_p, HEADS, 703u + t * 47u + (uint32_t)adversarial * 3u, adversarial);
            require_ok(ds4_gpu_tensor_write(gbeta_ref.t, 0, host_p,
                                            HEADS * sizeof(float)), "D beta write");
            /* the two low-rank rows the prologue expands */
            fill_activations(host_p, RANK, 704u + t * 53u + (uint32_t)adversarial * 3u, adversarial);
            require_ok(ds4_gpu_tensor_write(glf.t, 0, host_p, RANK * sizeof(float)), "D f_a write");
            fill_activations(host_p, RANK, 705u + t * 59u + (uint32_t)adversarial * 3u, adversarial);
            require_ok(ds4_gpu_tensor_write(glg.t, 0, host_p, RANK * sizeof(float)), "D g_a write");

            /* reference arm: explicit pair2in, then the two-dispatch core */
            guarded_poison(&ggate_ref);
            guarded_poison(&gogate_ref);
            guarded_poison(&gout_ref);
            require_ok(ds4_gpu_matmul_q8_0_pair2in_tensor(
                           ggate_ref.t, gogate_ref.t, g_model, MODEL_BYTES,
                           O_W_FB, O_W_GB, RANK, PROJECTION, glf.t, glg.t),
                       "D reference pair2in");
            require_ok(ds4_gpu_glm53_kda_decode_split2(
                           gout_ref.t, conv_ref, state_ref, scratch_ref,
                           gq.t, gk.t, gv.t, ggate_ref.t, gbeta_ref.t,
                           gogate_ref.t, g_model, MODEL_BYTES,
                           O_Q_CONV, O_K_CONV, O_V_CONV, O_A_LOG, O_DT_BIAS,
                           O_NORM, HEADS, 1, -5.0f, 1e-5f),
                       "D reference split2");

            /* candidate arm: the glue computes the gates itself */
            guarded_poison(&ggate_cand);
            guarded_poison(&gogate_cand);
            guarded_poison(&gout_cand);
            require_ok(ds4_gpu_glm53_kda_decode_glue(
                           gout_cand.t, conv_cand, state_cand, scratch_cand,
                           gq.t, gk.t, gv.t, ggate_cand.t, gbeta_ref.t,
                           gogate_cand.t, glf.t, glg.t, g_model, MODEL_BYTES,
                           O_Q_CONV, O_K_CONV, O_V_CONV, O_A_LOG, O_DT_BIAS,
                           O_NORM, O_W_FB, O_W_GB, RANK, /*lr_q8=*/1,
                           HEADS, 1, /*do_prologue=*/1, /*do_out=*/1,
                           -5.0f, 1e-5f),
                       "D candidate glue Q8 prologue + out fold");

            char what[112];
            snprintf(what, sizeof(what), "[%s] D: Q8 prologue, token %u, raw_gate (f_b)", set, t);
            guarded_read(&ggate_ref, host_a);
            guarded_read(&ggate_cand, host_b);
            require_identical(what, host_a, host_b, PROJECTION * sizeof(float));
            guarded_check_tail(&ggate_cand, what);

            snprintf(what, sizeof(what), "[%s] D: Q8 prologue, token %u, output_gate (g_b)", set, t);
            guarded_read(&gogate_ref, host_a);
            guarded_read(&gogate_cand, host_b);
            require_identical(what, host_a, host_b, PROJECTION * sizeof(float));
            guarded_check_tail(&gogate_cand, what);

            snprintf(what, sizeof(what), "[%s] D: Q8 prologue+out, token %u, output", set, t);
            guarded_read(&gout_ref, host_a);
            guarded_read(&gout_cand, host_b);
            require_identical(what, host_a, host_b, PROJECTION * sizeof(float));
            if (!adversarial) require_all_finite(what, host_a, PROJECTION);
            guarded_check_tail(&gout_cand, what);

            snprintf(what, sizeof(what), "[%s] D: Q8 prologue+out, token %u, conv ring", set, t);
            require_ok(ds4_gpu_tensor_read(conv_ref, 0, host_a, sizeof(host_a)), "conv ref read");
            require_ok(ds4_gpu_tensor_read(conv_cand, 0, host_b, sizeof(host_b)), "conv cand read");
            require_identical(what, host_a, host_b, sizeof(host_a));

            snprintf(what, sizeof(what), "[%s] D: Q8 prologue+out, token %u, delta-rule state", set, t);
            require_ok(ds4_gpu_tensor_read(state_ref, 0, state_a, sizeof(state_a)), "state ref read");
            require_ok(ds4_gpu_tensor_read(state_cand, 0, state_b, sizeof(state_b)), "state cand read");
            require_identical(what, state_a, state_b, sizeof(state_a));
        }
    }

    /* =================================================================
     * E. refusals must encode nothing
     * ================================================================= */
    {
        guarded_poison(&gout_cand);
        guarded_poison(&ggate_cand);
        guarded_poison(&gogate_cand);
        reset_state(conv_cand, state_cand);
        require_ok(ds4_gpu_tensor_read(conv_cand, 0, host_a, sizeof(host_a)),
                   "E conv baseline read");
        require_ok(ds4_gpu_tensor_read(state_cand, 0, state_a, sizeof(state_a)),
                   "E state baseline read");

        struct { const char *name; uint32_t rank; uint64_t fb, gb; int q8, pro, out; } bad[] = {
            { "E: Q8 rank 288 (nb=9 > NQ)", BAD_RANK, O_W_FB9, O_W_GB9, 1, 1, 1 },
            { "E: Q8 rank 127 (not a multiple of 32)", 127u, O_W_FB, O_W_GB, 1, 1, 1 },
            { "E: neither fold requested", RANK, O_W_FB, O_W_GB, 1, 0, 0 },
            { "E: BF16 rank 0", 0u, O_W_FB, O_W_GB, 0, 1, 1 },
        };
        for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
            const int rc = ds4_gpu_glm53_kda_decode_glue(
                gout_cand.t, conv_cand, state_cand, scratch_cand,
                gq.t, gk.t, gv.t, ggate_cand.t, gbeta_ref.t, gogate_cand.t,
                gbig_f.t, gbig_g.t, g_model, MODEL_BYTES,
                O_Q_CONV, O_K_CONV, O_V_CONV, O_A_LOG, O_DT_BIAS, O_NORM,
                bad[i].fb, bad[i].gb, bad[i].rank, bad[i].q8,
                HEADS, 1, bad[i].pro, bad[i].out, -5.0f, 1e-5f);
            if (rc != 0) {
                fprintf(stderr, "NOT REFUSED: %s returned %d\n", bad[i].name, rc);
                g_failures++;
                continue;
            }
            guarded_read(&gout_cand, host_b);
            for (int j = 0; j < PROJECTION; j++) {
                if (((const int32_t *)host_b)[j] != (int32_t)POISON) {
                    fprintf(stderr, "%s refused but wrote out[%d]\n", bad[i].name, j);
                    g_failures++;
                    break;
                }
            }
            require_ok(ds4_gpu_tensor_read(conv_cand, 0, host_b, sizeof(host_b)),
                       "E conv read");
            require_identical(bad[i].name, host_a, host_b, sizeof(host_a));
            require_ok(ds4_gpu_tensor_read(state_cand, 0, state_b, sizeof(state_b)),
                       "E state read");
            require_identical(bad[i].name, state_a, state_b, sizeof(state_a));
        }

        /* the fold host must reject an extent that is not a multiple of nr0 */
        {
            ds4_gpu_tensor *fused[6] = { gq_fus.t, gk_fus.t, gv_fus.t,
                                         glf_fus.t, gbeta_fus.t, glg_fus.t };
            const uint64_t offs[6] = { O_W_Q, O_W_K, O_W_V,
                                       O_W_FA, O_W_BETA, O_W_GA };
            const uint64_t odd[6]  = { PROJECTION, PROJECTION, PROJECTION,
                                       RANK, 3u /* odd */, RANK };
            guarded_poison(&gbeta_fus);
            if (ds4_gpu_matmul_q8_0_kda6_flat_tensor(
                    fused, g_model, MODEL_BYTES, offs, EMBD, odd, gx.t) != 0) {
                fprintf(stderr, "NOT REFUSED: kda6_flat accepted an odd extent\n");
                g_failures++;
            }
            guarded_read(&gbeta_fus, host_b);
            if (((const int32_t *)host_b)[0] != (int32_t)POISON) {
                fprintf(stderr, "kda6_flat refused but wrote its output\n");
                g_failures++;
            }
        }
    }

    if (g_failures != 0) {
        fprintf(stderr, "test_glm53_kda_q8: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test_glm53_kda_q8: all KDA Q8_0 paths bit-identical "
           "(outputs, conv ring, delta-rule state), refusals encode nothing\n");
    return 0;
}
