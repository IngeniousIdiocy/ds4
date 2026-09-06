/* Chained-dispatch harness for the GLM-5.3 decode "small fusion" campaign.
 *
 * Runs the REAL repo kernels through the real ds4_gpu_* entry points (the Metal
 * library is compiled from the metal source directory at runtime, so run this
 * from the repo root).  Each fusion gets two things:
 *
 *   identity  - >= 50,000 randomized cases, old path and new path in the SAME
 *               process against the same inputs, compared bit for bit.
 *   timing    - best-of-5 x 200 chained dispatches inside one command buffer on
 *               the serial encoder, which is the ordering the decode graph uses,
 *               so the measured delta is the dependent-dispatch turnaround the
 *               fusion removes.
 *
 * Build:  make metal-small-fuse-bench
 * Usage:  ./speed-bench/metal_small_fuse_bench [qakv|router|all]
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

#define REPS 200
#define ROUNDS 5

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rng_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 32);
}
static float rng_f32(void) {
    /* Wide dynamic range so cancellation and reassociation would show up. */
    return ((float)(rng_u32() & 0xffffffu) / 8388608.0f - 1.0f) *
           (1.0f + (float)(rng_u32() & 7u));
}

static void map_model(const uint8_t *model, uint64_t bytes, uint64_t max_tensor) {
    if (!ds4_gpu_set_model_map_range(model, bytes, 0, bytes, max_tensor)) {
        fprintf(stderr, "small-fuse bench: ds4_gpu_set_model_map_range failed\n");
        exit(1);
    }
}

static void die(const char *what) {
    fprintf(stderr, "small-fuse bench: %s failed\n", what);
    exit(1);
}

/* ---------------------------------------------------------------- Q8_0 layout */

typedef struct {
    uint16_t d;      /* half */
    int8_t   qs[32];
} blk_q8_0;

static uint16_t f32_to_f16(float value) {
    /* Only used to fill synthetic weights; small magnitudes, no denormals. */
    union { float f; uint32_t u; } in = { .f = value };
    const uint32_t sign = (in.u >> 31) & 1u;
    int32_t exp = (int32_t)((in.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = in.u & 0x7fffffu;
    if (exp <= 0) return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | (31u << 10));
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (mant >> 13));
}

static void fill_q8_rows(uint8_t *base, uint64_t rows, uint64_t in_dim) {
    const uint64_t nblk = in_dim / 32u;
    for (uint64_t r = 0; r < rows; r++) {
        blk_q8_0 *row = (blk_q8_0 *)(base + r * nblk * sizeof(blk_q8_0));
        for (uint64_t b = 0; b < nblk; b++) {
            row[b].d = f32_to_f16(0.002f + (float)(rng_u32() & 255u) * 0.0001f);
            for (int i = 0; i < 32; i++) {
                row[b].qs[i] = (int8_t)(int32_t)((rng_u32() & 255u) - 128u);
            }
        }
    }
}

/* ------------------------------------------------------------------ fusion A */

/* GLM-5.3 DSA decode shapes. */
#define A_IN        4096u
#define A_OUT0      1536u   /* attn_q_a       */
#define A_OUT1       512u   /* attn_kv_a_mqa  */

static uint64_t q8_row_bytes(uint64_t in_dim) { return (in_dim / 32u) * 34u; }

static int qakv_identity(void) {
    /* Small randomized shapes: 50,000 cases at decode-realistic strides but
     * cheap enough to run old-vs-new in one process, plus a real-shape tail. */
    enum { CASES_SMALL = 50000, CASES_REAL = 200 };
    const uint64_t model_bytes = 1u << 24;
    uint8_t *model = mmap(NULL, model_bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) die("mmap");
    map_model(model, model_bytes, q8_row_bytes(A_IN) * A_OUT0);

    /* Buffers are sized for the real DSA shape; the randomized cases use the
     * smaller ranges below so 50,000 of them stay cheap. */
    const uint32_t max_in = A_IN;
    const uint32_t max_out = A_OUT0;
    const uint32_t rnd_in = 1024u;
    const uint32_t rnd_out = 256u;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)max_in * sizeof(float));
    ds4_gpu_tensor *ref0 = ds4_gpu_tensor_alloc((uint64_t)max_out * sizeof(float));
    ds4_gpu_tensor *ref1 = ds4_gpu_tensor_alloc((uint64_t)max_out * sizeof(float));
    ds4_gpu_tensor *new0 = ds4_gpu_tensor_alloc((uint64_t)max_out * sizeof(float));
    ds4_gpu_tensor *new1 = ds4_gpu_tensor_alloc((uint64_t)max_out * sizeof(float));
    if (!x || !ref0 || !ref1 || !new0 || !new1) die("tensor_alloc");

    float *xh = malloc((size_t)max_in * sizeof(float));
    uint32_t *r0 = malloc((size_t)max_out * sizeof(uint32_t));
    uint32_t *r1 = malloc((size_t)max_out * sizeof(uint32_t));
    uint32_t *n0 = malloc((size_t)max_out * sizeof(uint32_t));
    uint32_t *n1 = malloc((size_t)max_out * sizeof(uint32_t));
    if (!xh || !r0 || !r1 || !n0 || !n1) die("malloc");

    uint64_t bad = 0, checked = 0, skipped = 0;
    for (long c = 0; c < CASES_SMALL + CASES_REAL; c++) {
        const bool real_shape = c >= CASES_SMALL;
        uint32_t in_dim, out0, out1;
        if (real_shape) {
            in_dim = A_IN; out0 = A_OUT0; out1 = A_OUT1;
        } else {
            in_dim = 32u * (1u + (rng_u32() % (rnd_in / 32u)));
            out0 = 2u * (1u + (rng_u32() % (rnd_out / 2u)));
            out1 = 2u * (1u + (rng_u32() % (rnd_out / 2u)));
        }
        const uint64_t rb = q8_row_bytes(in_dim);
        const uint64_t w0_bytes = rb * out0;
        const uint64_t w1_bytes = rb * out1;
        if (w0_bytes + w1_bytes > model_bytes) { skipped++; continue; }

        fill_q8_rows(model, out0, in_dim);
        fill_q8_rows(model + w0_bytes, out1, in_dim);
        for (uint32_t i = 0; i < in_dim; i++) xh[i] = rng_f32();
        if (!ds4_gpu_tensor_write(x, 0, xh, (uint64_t)in_dim * sizeof(float))) {
            die("tensor_write");
        }

        if (!ds4_gpu_begin_commands()) die("begin_commands");
        if (!ds4_gpu_matmul_q8_0_tensor(ref0, model, model_bytes, 0,
                                        in_dim, out0, x, 1)) die("ref q_a");
        if (!ds4_gpu_matmul_q8_0_tensor(ref1, model, model_bytes, w0_bytes,
                                        in_dim, out1, x, 1)) die("ref kv_a");
        if (!ds4_gpu_matmul_q8_0_pair_flat_tensor(new0, new1, model, model_bytes,
                                                 0, w0_bytes, in_dim,
                                                 out0, out1, x)) die("fused pair");
        if (!ds4_gpu_end_commands()) die("end_commands");
        if (!ds4_gpu_synchronize()) die("synchronize");

        if (!ds4_gpu_tensor_read(ref0, 0, r0, (uint64_t)out0 * 4) ||
            !ds4_gpu_tensor_read(ref1, 0, r1, (uint64_t)out1 * 4) ||
            !ds4_gpu_tensor_read(new0, 0, n0, (uint64_t)out0 * 4) ||
            !ds4_gpu_tensor_read(new1, 0, n1, (uint64_t)out1 * 4)) {
            die("tensor_read");
        }
        for (uint32_t i = 0; i < out0; i++) if (r0[i] != n0[i]) bad++;
        for (uint32_t i = 0; i < out1; i++) if (r1[i] != n1[i]) bad++;
        checked += out0 + out1;
        if (bad) {
            fprintf(stderr,
                    "qakv identity: MISMATCH at case %ld (in=%u out0=%u out1=%u)\n",
                    c, in_dim, out0, out1);
            break;
        }
    }
    printf("qakv identity: cases=%ld skipped=%llu values=%llu differing_bits=%llu\n",
           (long)(CASES_SMALL + CASES_REAL), (unsigned long long)skipped,
           (unsigned long long)checked, (unsigned long long)bad);
    return bad == 0 ? 0 : 1;
}

static double qakv_time(int mode) {  /* 0 separate, 1 flat, 2 max-extent pair */
    const uint64_t rb = q8_row_bytes(A_IN);
    const uint64_t w0_bytes = rb * A_OUT0;
    const uint64_t w1_bytes = rb * A_OUT1;
    const uint64_t model_bytes = (w0_bytes + w1_bytes + 4095u) & ~4095ull;
    uint8_t *model = mmap(NULL, model_bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) die("mmap");
    fill_q8_rows(model, A_OUT0, A_IN);
    fill_q8_rows(model + w0_bytes, A_OUT1, A_IN);
    map_model(model, model_bytes, w0_bytes);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)A_IN * sizeof(float));
    ds4_gpu_tensor *o0 = ds4_gpu_tensor_alloc((uint64_t)A_OUT0 * sizeof(float));
    ds4_gpu_tensor *o1 = ds4_gpu_tensor_alloc((uint64_t)A_OUT1 * sizeof(float));
    if (!x || !o0 || !o1) die("tensor_alloc");
    float *xh = malloc((size_t)A_IN * sizeof(float));
    for (uint32_t i = 0; i < A_IN; i++) xh[i] = rng_f32();
    if (!ds4_gpu_tensor_write(x, 0, xh, (uint64_t)A_IN * sizeof(float))) {
        die("tensor_write");
    }

    double best = 1e30;
    for (int round = 0; round < ROUNDS + 1; round++) {
        const double t0 = now_ms();
        if (!ds4_gpu_begin_commands()) die("begin_commands");
        for (int i = 0; i < REPS; i++) {
            if (mode == 1) {
                if (!ds4_gpu_matmul_q8_0_pair_flat_tensor(o0, o1, model,
                                                          model_bytes, 0,
                                                          w0_bytes, A_IN,
                                                          A_OUT0, A_OUT1, x)) {
                    die("fused pair");
                }
            } else if (mode == 2) {
                if (!ds4_gpu_matmul_q8_0_pair_tensor(o0, o1, model, model_bytes,
                                                     0, w0_bytes, A_IN,
                                                     A_OUT0, A_OUT1, x, 1)) {
                    die("max-extent pair");
                }
            } else {
                if (!ds4_gpu_matmul_q8_0_tensor(o0, model, model_bytes, 0,
                                                A_IN, A_OUT0, x, 1)) die("q_a");
                if (!ds4_gpu_matmul_q8_0_tensor(o1, model, model_bytes, w0_bytes,
                                                A_IN, A_OUT1, x, 1)) die("kv_a");
            }
        }
        if (!ds4_gpu_end_commands()) die("end_commands");
        if (!ds4_gpu_synchronize()) die("synchronize");
        const double us = (now_ms() - t0) * 1000.0 / (double)REPS;
        if (round > 0 && us < best) best = us;
    }
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(o0);
    ds4_gpu_tensor_free(o1);
    free(xh);
    munmap(model, model_bytes);
    return best;
}

/* ------------------------------------------------------------------ fusion C */

#define C_IN      4096u
#define C_EXPERT   288u
#define C_USED       8u

static int router_identity(void) {
    enum { CASES = 50000 };
    /* Model layout: [0] router weight (f32, C_EXPERT x C_IN), then bias. */
    const uint64_t w_bytes = (uint64_t)C_EXPERT * C_IN * sizeof(float);
    const uint64_t bias_off = w_bytes;
    const uint64_t model_bytes =
        (w_bytes + (uint64_t)C_EXPERT * sizeof(float) + 4095u) & ~4095ull;
    uint8_t *model = mmap(NULL, model_bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) die("mmap");
    map_model(model, model_bytes, w_bytes);
    float *bias = (float *)(model + bias_off);

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)C_IN * sizeof(float));
    ds4_gpu_tensor *logits_a = ds4_gpu_tensor_alloc((uint64_t)C_EXPERT * 4);
    ds4_gpu_tensor *logits_b = ds4_gpu_tensor_alloc((uint64_t)C_EXPERT * 4);
    ds4_gpu_tensor *sel_a = ds4_gpu_tensor_alloc((uint64_t)C_USED * 4);
    ds4_gpu_tensor *sel_b = ds4_gpu_tensor_alloc((uint64_t)C_USED * 4);
    ds4_gpu_tensor *wt_a = ds4_gpu_tensor_alloc((uint64_t)C_USED * 4);
    ds4_gpu_tensor *wt_b = ds4_gpu_tensor_alloc((uint64_t)C_USED * 4);
    ds4_gpu_tensor *pr_a = ds4_gpu_tensor_alloc((uint64_t)C_EXPERT * 4);
    ds4_gpu_tensor *pr_b = ds4_gpu_tensor_alloc((uint64_t)C_EXPERT * 4);
    ds4_gpu_tensor *counter = ds4_gpu_tensor_alloc(sizeof(uint32_t));
    if (!xt || !logits_a || !logits_b || !sel_a || !sel_b || !wt_a || !wt_b ||
        !pr_a || !pr_b || !counter) {
        die("tensor_alloc");
    }
    const uint32_t zero = 0;
    if (!ds4_gpu_tensor_write(counter, 0, &zero, sizeof(zero))) die("counter init");

    float *xh = malloc((size_t)C_IN * sizeof(float));
    uint32_t la[C_EXPERT], lb[C_EXPERT], pa[C_EXPERT], pb[C_EXPERT];
    uint32_t sa[C_USED], sb[C_USED], wa[C_USED], wb[C_USED];
    if (!xh) die("malloc");

    uint64_t bad = 0, checked = 0;
    for (long c = 0; c < CASES; c++) {
        /* Fresh weights every 250 cases keeps the logits distribution moving
         * without paying 4.7 MB of host fill per case. */
        if ((c % 250) == 0) {
            float *w = (float *)model;
            for (uint64_t i = 0; i < (uint64_t)C_EXPERT * C_IN; i++) {
                w[i] = rng_f32() * 0.02f;
            }
            for (uint32_t i = 0; i < C_EXPERT; i++) bias[i] = rng_f32() * 0.5f;
        }
        for (uint32_t i = 0; i < C_IN; i++) xh[i] = rng_f32();
        /* Every 7th case forces heavy score ties, which is where a different
         * reduction shape would diverge on the tie-break. */
        if ((c % 7) == 0) {
            for (uint32_t i = 0; i < C_EXPERT; i++) {
                bias[i] = (float)(i % 3) * 0.25f;
            }
            for (uint32_t i = 0; i < C_IN; i++) xh[i] = 0.0f;
        }
        if (!ds4_gpu_tensor_write(xt, 0, xh, (uint64_t)C_IN * sizeof(float))) {
            die("tensor_write");
        }

        if (!ds4_gpu_begin_commands()) die("begin_commands");
        if (!ds4_gpu_matmul_f32_tensor(logits_a, model, model_bytes, 0,
                                       C_IN, C_EXPERT, xt, 1)) die("ref logits");
        if (!ds4_gpu_glm_router_select_tensor(sel_a, wt_a, pr_a, model,
                                              model_bytes, bias_off, logits_a,
                                              C_EXPERT, C_USED, 2.5f)) {
            die("ref select");
        }
        if (!ds4_gpu_glm_router_logits_select_tail_tensor(
                    logits_b, sel_b, wt_b, pr_b, counter, model, model_bytes,
                    0, bias_off, C_IN, C_EXPERT, C_USED, 2.5f, xt)) {
            die("fused logits+select");
        }
        if (!ds4_gpu_end_commands()) die("end_commands");
        if (!ds4_gpu_synchronize()) die("synchronize");

        if (!ds4_gpu_tensor_read(logits_a, 0, la, sizeof(la)) ||
            !ds4_gpu_tensor_read(logits_b, 0, lb, sizeof(lb)) ||
            !ds4_gpu_tensor_read(pr_a, 0, pa, sizeof(pa)) ||
            !ds4_gpu_tensor_read(pr_b, 0, pb, sizeof(pb)) ||
            !ds4_gpu_tensor_read(sel_a, 0, sa, sizeof(sa)) ||
            !ds4_gpu_tensor_read(sel_b, 0, sb, sizeof(sb)) ||
            !ds4_gpu_tensor_read(wt_a, 0, wa, sizeof(wa)) ||
            !ds4_gpu_tensor_read(wt_b, 0, wb, sizeof(wb))) {
            die("tensor_read");
        }
        uint32_t d_log = 0, d_prob = 0, d_sel = 0, d_wt = 0;
        for (uint32_t i = 0; i < C_EXPERT; i++) {
            if (la[i] != lb[i]) d_log++;
            if (pa[i] != pb[i]) d_prob++;
        }
        for (uint32_t i = 0; i < C_USED; i++) {
            if (sa[i] != sb[i]) d_sel++;
            if (wa[i] != wb[i]) d_wt++;
        }
        bad += d_log + d_prob + d_sel + d_wt;
        if (d_log + d_prob + d_sel + d_wt) {
            fprintf(stderr, "  case %ld: logits=%u probs=%u sel=%u wt=%u\n",
                    c, d_log, d_prob, d_sel, d_wt);
        }
        checked += 2u * C_EXPERT + 2u * C_USED;
        if (bad) {
            fprintf(stderr, "router identity: MISMATCH at case %ld\n", c);
            break;
        }
    }
    printf("router identity: cases=%ld values=%llu differing_bits=%llu\n",
           (long)CASES, (unsigned long long)checked, (unsigned long long)bad);
    return bad == 0 ? 0 : 1;
}

static double router_time(int mode) {  /* 0 both, 1 fused, 2 logits only, 3 select only */
    const uint64_t w_bytes = (uint64_t)C_EXPERT * C_IN * sizeof(float);
    const uint64_t bias_off = w_bytes;
    const uint64_t model_bytes =
        (w_bytes + (uint64_t)C_EXPERT * sizeof(float) + 4095u) & ~4095ull;
    uint8_t *model = mmap(NULL, model_bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) die("mmap");
    float *w = (float *)model;
    for (uint64_t i = 0; i < (uint64_t)C_EXPERT * C_IN; i++) w[i] = rng_f32() * 0.02f;
    float *bias = (float *)(model + bias_off);
    for (uint32_t i = 0; i < C_EXPERT; i++) bias[i] = rng_f32() * 0.5f;
    map_model(model, model_bytes, w_bytes);

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)C_IN * sizeof(float));
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc((uint64_t)C_EXPERT * 4);
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc((uint64_t)C_USED * 4);
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc((uint64_t)C_USED * 4);
    ds4_gpu_tensor *pr = ds4_gpu_tensor_alloc((uint64_t)C_EXPERT * 4);
    ds4_gpu_tensor *counter = ds4_gpu_tensor_alloc(sizeof(uint32_t));
    if (!xt || !logits || !sel || !wt || !pr || !counter) die("tensor_alloc");
    const uint32_t zero = 0;
    if (!ds4_gpu_tensor_write(counter, 0, &zero, sizeof(zero))) die("counter init");
    float *xh = malloc((size_t)C_IN * sizeof(float));
    for (uint32_t i = 0; i < C_IN; i++) xh[i] = rng_f32();
    if (!ds4_gpu_tensor_write(xt, 0, xh, (uint64_t)C_IN * sizeof(float))) {
        die("tensor_write");
    }

    double best = 1e30;
    for (int round = 0; round < ROUNDS + 1; round++) {
        const double t0 = now_ms();
        if (!ds4_gpu_begin_commands()) die("begin_commands");
        for (int i = 0; i < REPS; i++) {
            if (mode == 1) {
                if (!ds4_gpu_glm_router_logits_select_tail_tensor(
                            logits, sel, wt, pr, counter, model, model_bytes,
                            0, bias_off, C_IN, C_EXPERT, C_USED, 2.5f, xt)) {
                    die("fused logits+select");
                }
            } else {
                if (mode != 3 &&
                    !ds4_gpu_matmul_f32_tensor(logits, model, model_bytes, 0,
                                               C_IN, C_EXPERT, xt, 1)) die("logits");
                if (mode != 2 &&
                    !ds4_gpu_glm_router_select_tensor(sel, wt, pr, model,
                                                      model_bytes, bias_off,
                                                      logits, C_EXPERT, C_USED,
                                                      2.5f)) die("select");
            }
        }
        if (!ds4_gpu_end_commands()) die("end_commands");
        if (!ds4_gpu_synchronize()) die("synchronize");
        const double us = (now_ms() - t0) * 1000.0 / (double)REPS;
        if (round > 0 && us < best) best = us;
    }
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(sel);
    ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(pr);
    ds4_gpu_tensor_free(counter);
    free(xh);
    munmap(model, model_bytes);
    return best;
}

/* ---------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    const char *which = argc > 1 ? argv[1] : "all";
    if (!ds4_gpu_init()) die("ds4_gpu_init");
    int rc = 0;

    if (!strcmp(which, "all") || !strcmp(which, "qakv")) {
        const double sep = qakv_time(0);
        const double fus = qakv_time(1);
        const double mx = qakv_time(2);
        printf("qakv timing: separate=%.2f us flat_fused=%.2f us max_extent_fused=%.2f us "
               "delta=%.2f us (%.3fx)\n",
               sep, fus, mx, sep - fus, sep / fus);
        rc |= qakv_identity();
    }
    if (!strcmp(which, "all") || !strcmp(which, "router")) {
        const double sep = router_time(0);
        const double fus = router_time(1);
        const double lg = router_time(2);
        const double sl = router_time(3);
        printf("router timing: logits_only=%.2f us select_only=%.2f us separate=%.2f us "
               "fused=%.2f us delta=%.2f us (%.3fx)\n",
               lg, sl, sep, fus, sep - fus, sep / fus);
        rc |= router_identity();
    }
    return rc;
}
