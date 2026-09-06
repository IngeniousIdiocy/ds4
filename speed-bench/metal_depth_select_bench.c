/*
 * metal_depth_select_bench - chained microbench + identity gauntlet for the
 * GLM-5.3 DSA indexer scoring / top-k selection / pool-expansion chain.
 *
 * The decode graph runs, per DSA layer and per token:
 *
 *   ds4_gpu_glm_indexer_score_one_tensor   (pooled F16 keys -> f32 scores)
 *   ds4_gpu_indexer_topk_tensor            (top-selected_pools of n_comp)
 *   ds4_gpu_glm53_expand_pool_selection_tensor
 *
 * All three land in one serial compute encoder inside one command buffer, so
 * this bench replays exactly that: begin_commands(), N tokens x 11 layers of
 * dependent dispatches over the same buffers, flush_commands().
 *
 * Usage:
 *   metal_depth_select_bench bench   [--pos N,...] [--tokens N] [--reps N]
 *   metal_depth_select_bench identity-topk  [--cases N] [--pos N,...]
 *   metal_depth_select_bench identity-score [--cases N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "ds4_gpu.h"

/* ds4_metal.o references this logging helper, which lives in ds4.c; the bench
 * deliberately links only the Metal backend. */
int ds4_gpu_commands_active(void);
int ds4_log_is_tty(FILE *fp);
int ds4_log_is_tty(FILE *fp) { (void)fp; return 0; }

/* -ffast-math rejects the NAN/INFINITY macros, so build the bit patterns. */
static float f32_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
#define DS4_BENCH_INF     f32_bits(0x7f800000u)
#define DS4_BENCH_NEG_INF f32_bits(0xff800000u)
#define DS4_BENCH_NAN     f32_bits(0x7fc00000u)

#define N_DSA_LAYERS       11u
#define N_INDEXER_HEAD     32u
#define N_INDEXER_HEAD_DIM 128u
#define POOL_SIZE          4u
#define INDEX_TOPK         2048u
#define SELECTED_LIMIT     2051u

/* ---------------------------------------------------------------- utility */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static uint64_t rng_state = 0x853c49e6748fea9bULL;

static uint32_t rng_u32(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_state >> 33);
}

static float rng_f32_unit(void) {
    return (float)(rng_u32() >> 8) * (1.0f / 16777216.0f);
}

/* half<->float without depending on _Float16 codegen details */
static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } v = { .f = f };
    uint32_t u = v.u;
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xff) - 127 + 15;
    uint32_t man = u & 0x7fffffu;
    if (((u >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_man = man >> shift;
        if ((man >> (shift - 1)) & 1u) half_man++;
        return (uint16_t)(sign | half_man);
    }
    uint32_t half_man = man >> 13;
    if ((man >> 12) & 1u) {
        half_man++;
        if (half_man == 0x400u) { half_man = 0; exp++; if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_man);
}

/* ------------------------------------------------------------------ state */

typedef struct {
    uint32_t n_comp;
    uint32_t selected_pools;
    uint32_t pos;

    ds4_gpu_tensor *q;
    ds4_gpu_tensor *weights;
    ds4_gpu_tensor *keys;
    ds4_gpu_tensor *scores;
    ds4_gpu_tensor *pool_selected;
    ds4_gpu_tensor *raw_selected;
} chain_t;

static const float INDEXER_SCALE_DENOM_HEADS = (float)(N_INDEXER_HEAD_DIM * N_INDEXER_HEAD);

static float indexer_scale(void) { return 1.0f / sqrtf(INDEXER_SCALE_DENOM_HEADS); }

static int chain_alloc(chain_t *c, uint32_t pos) {
    memset(c, 0, sizeof(*c));
    c->pos = pos;
    c->n_comp = pos / POOL_SIZE;
    if (c->n_comp == 0) return 0;
    c->selected_pools = INDEX_TOPK / POOL_SIZE;
    if (c->selected_pools > c->n_comp) c->selected_pools = c->n_comp;

    c->q       = ds4_gpu_tensor_alloc((uint64_t)N_INDEXER_HEAD * N_INDEXER_HEAD_DIM * sizeof(float));
    c->weights = ds4_gpu_tensor_alloc((uint64_t)N_INDEXER_HEAD * sizeof(float));
    c->keys    = ds4_gpu_tensor_alloc((uint64_t)c->n_comp * N_INDEXER_HEAD_DIM * sizeof(uint16_t));
    c->scores  = ds4_gpu_tensor_alloc((uint64_t)c->n_comp * sizeof(float));
    c->pool_selected = ds4_gpu_tensor_alloc((uint64_t)c->selected_pools * sizeof(uint32_t));
    c->raw_selected  = ds4_gpu_tensor_alloc((uint64_t)SELECTED_LIMIT * sizeof(uint32_t));
    if (!c->q || !c->weights || !c->keys || !c->scores || !c->pool_selected || !c->raw_selected) {
        fprintf(stderr, "alloc failed at pos=%u\n", pos);
        return 0;
    }

    float *qh = malloc((size_t)N_INDEXER_HEAD * N_INDEXER_HEAD_DIM * sizeof(float));
    float *wh = malloc((size_t)N_INDEXER_HEAD * sizeof(float));
    uint16_t *kh = malloc((size_t)c->n_comp * N_INDEXER_HEAD_DIM * sizeof(uint16_t));
    if (!qh || !wh || !kh) return 0;
    for (uint32_t i = 0; i < N_INDEXER_HEAD * N_INDEXER_HEAD_DIM; i++)
        qh[i] = (rng_f32_unit() - 0.5f) * 2.0f;
    for (uint32_t i = 0; i < N_INDEXER_HEAD; i++) wh[i] = rng_f32_unit();
    for (uint64_t i = 0; i < (uint64_t)c->n_comp * N_INDEXER_HEAD_DIM; i++)
        kh[i] = f32_to_f16((rng_f32_unit() - 0.5f) * 2.0f);
    ds4_gpu_tensor_write(c->q, 0, qh, (uint64_t)N_INDEXER_HEAD * N_INDEXER_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor_write(c->weights, 0, wh, (uint64_t)N_INDEXER_HEAD * sizeof(float));
    ds4_gpu_tensor_write(c->keys, 0, kh, (uint64_t)c->n_comp * N_INDEXER_HEAD_DIM * sizeof(uint16_t));
    free(qh); free(wh); free(kh);
    return 1;
}

static void chain_free(chain_t *c) {
    ds4_gpu_tensor_free(c->q);
    ds4_gpu_tensor_free(c->weights);
    ds4_gpu_tensor_free(c->keys);
    ds4_gpu_tensor_free(c->scores);
    ds4_gpu_tensor_free(c->pool_selected);
    ds4_gpu_tensor_free(c->raw_selected);
    memset(c, 0, sizeof(*c));
}

/* Score distributions designed to stress the selection comparator. */
static void fill_scores(float *v, uint32_t n, uint32_t mode) {
    switch (mode % 8u) {
    case 0:  /* smooth random, few ties */
        for (uint32_t i = 0; i < n; i++) v[i] = rng_f32_unit();
        break;
    case 1:  /* heavy ties: tiny alphabet */
        for (uint32_t i = 0; i < n; i++) v[i] = (float)(rng_u32() % 4u);
        break;
    case 2:  /* all identical */
        for (uint32_t i = 0; i < n; i++) v[i] = 0.5f;
        break;
    case 3:  /* mostly zero with a plateau exactly at the cut */
        for (uint32_t i = 0; i < n; i++) v[i] = (rng_u32() & 1u) ? 1.0f : 0.0f;
        break;
    case 4:  /* denormals and tiny values */
        for (uint32_t i = 0; i < n; i++) {
            union { uint32_t u; float f; } b = { .u = rng_u32() & 0x000fffffu };
            v[i] = b.f;
        }
        break;
    case 5:  /* +-0 mixed with zeros */
        for (uint32_t i = 0; i < n; i++) {
            uint32_t r = rng_u32() % 3u;
            v[i] = r == 0 ? 0.0f : (r == 1 ? -0.0f : 1.0f);
        }
        break;
    case 6:  /* infinities */
        for (uint32_t i = 0; i < n; i++) {
            uint32_t r = rng_u32() % 4u;
            v[i] = r == 0 ? DS4_BENCH_INF : (r == 1 ? DS4_BENCH_NEG_INF : rng_f32_unit());
        }
        break;
    default: /* wide exponent spread */
        for (uint32_t i = 0; i < n; i++) {
            union { uint32_t u; float f; } b = { .u = (rng_u32() & 0x7f7fffffu) };
            v[i] = b.f * ((rng_u32() & 1u) ? 1.0f : -1.0f);
        }
        break;
    }
}

/* stage bitmask */
#define ST_SCORE  1u
#define ST_TOPK   2u
#define ST_EXPAND 4u
#define ST_ALL    7u

static int chain_encode_layer(chain_t *c, unsigned stages) {
    if (stages & ST_SCORE) {
        if (!ds4_gpu_glm_indexer_score_one_tensor(c->scores, c->q, c->weights, c->keys,
                                                  c->n_comp, N_INDEXER_HEAD, N_INDEXER_HEAD_DIM,
                                                  indexer_scale(), true)) return 0;
    }
    if ((stages & ST_TOPK) && (stages & ST_EXPAND) &&
        ds4_gpu_glm53_indexer_topk_expand_tensor(c->raw_selected, c->pool_selected, c->scores,
                                                 c->n_comp, 1, c->selected_pools, c->pos,
                                                 INDEX_TOPK, POOL_SIZE, SELECTED_LIMIT)) {
        return 1;   /* selection + pool expansion in one dispatch pair */
    }
    if (stages & ST_TOPK) {
        if (!ds4_gpu_indexer_topk_tensor(c->pool_selected, c->scores, c->n_comp, 1,
                                         c->selected_pools)) return 0;
    }
    if (stages & ST_EXPAND) {
        if (!ds4_gpu_glm53_expand_pool_selection_tensor(c->raw_selected, c->pool_selected, 1,
                                                        c->pos, c->selected_pools, INDEX_TOPK,
                                                        POOL_SIZE, SELECTED_LIMIT)) return 0;
    }
    return 1;
}

/* Fused topk+expand vs the two-dispatch sequence: raw_selected must match. */
static int cmd_identity_expand(int argc, char **argv) {
    uint32_t cases = 2000;
    uint32_t positions[4] = { 8192u, 62234u, 100000u, 32768u };
    for (int i = 0; i < argc; i++)
        if (!strcmp(argv[i], "--cases") && i + 1 < argc) cases = (uint32_t)strtoul(argv[++i], NULL, 10);

    uint64_t checked = 0, bad = 0;
    for (uint32_t p = 0; p < 4; p++) {
        chain_t c;
        if (!chain_alloc(&c, positions[p])) return 1;
        ds4_gpu_tensor *raw_ref = ds4_gpu_tensor_alloc((uint64_t)SELECTED_LIMIT * sizeof(uint32_t));
        float    *hs = malloc((size_t)c.n_comp * sizeof(float));
        uint32_t *ha = malloc((size_t)SELECTED_LIMIT * sizeof(uint32_t));
        uint32_t *hb = malloc((size_t)SELECTED_LIMIT * sizeof(uint32_t));
        if (!raw_ref || !hs || !ha || !hb) return 1;
        for (uint32_t ci = 0; ci < cases; ci++) {
            fill_scores(hs, c.n_comp, ci);
            ds4_gpu_tensor_write(c.scores, 0, hs, (uint64_t)c.n_comp * sizeof(float));

            ds4_gpu_glm_indexer_select_override(-1, 0);
            if (!ds4_gpu_indexer_topk_tensor(c.pool_selected, c.scores, c.n_comp, 1,
                                             c.selected_pools)) return 1;
            if (!ds4_gpu_glm53_expand_pool_selection_tensor(raw_ref, c.pool_selected, 1, c.pos,
                                                            c.selected_pools, INDEX_TOPK,
                                                            POOL_SIZE, SELECTED_LIMIT)) return 1;
            ds4_gpu_tensor_read(raw_ref, 0, ha, (uint64_t)SELECTED_LIMIT * sizeof(uint32_t));

            ds4_gpu_glm_indexer_select_override(-1, 1);
            if (!ds4_gpu_glm53_indexer_topk_expand_tensor(c.raw_selected, c.pool_selected, c.scores,
                                                          c.n_comp, 1, c.selected_pools, c.pos,
                                                          INDEX_TOPK, POOL_SIZE, SELECTED_LIMIT)) {
                fprintf(stderr, "fused topk+expand refused at pos=%u\n", c.pos);
                return 1;
            }
            ds4_gpu_tensor_read(c.raw_selected, 0, hb, (uint64_t)SELECTED_LIMIT * sizeof(uint32_t));

            checked++;
            if (memcmp(ha, hb, SELECTED_LIMIT * sizeof(uint32_t)) != 0) {
                if (!bad) {
                    uint32_t j = 0;
                    while (j < SELECTED_LIMIT && ha[j] == hb[j]) j++;
                    fprintf(stderr, "EXPAND MISMATCH pos=%u case=%u slot=%u ref=%u fused=%u\n",
                            c.pos, ci, j, ha[j], hb[j]);
                }
                bad++;
            }
        }
        free(hs); free(ha); free(hb);
        ds4_gpu_tensor_free(raw_ref);
        chain_free(&c);
    }
    ds4_gpu_glm_indexer_select_override(-1, -1);
    printf("identity-expand: cases=%llu mismatches=%llu\n",
           (unsigned long long)checked, (unsigned long long)bad);
    return bad != 0;
}

/* returns best (min) us per token over `reps` runs of `tokens` tokens */
static double bench_stages(chain_t *c, unsigned stages, uint32_t tokens, uint32_t reps) {
    double best = 1e30;
    for (uint32_t r = 0; r < reps + 1; r++) {   /* first rep = warmup */
        if (!ds4_gpu_commands_active() && !ds4_gpu_begin_commands()) {
            fprintf(stderr, "begin_commands failed\n"); return -1;
        }
        double t0 = now_us();
        for (uint32_t t = 0; t < tokens; t++)
            for (uint32_t l = 0; l < N_DSA_LAYERS; l++)
                if (!chain_encode_layer(c, stages)) { fprintf(stderr, "encode failed\n"); return -1; }
        /* synchronize() commits the open batch and blocks until the GPU is
         * idle; flush_commands() alone only commits. */
        if (!ds4_gpu_synchronize()) { fprintf(stderr, "synchronize failed\n"); return -1; }
        double dt = now_us() - t0;
        if (r > 0 && dt < best) best = dt;
    }
    return best / (double)tokens;
}

/* ------------------------------------------------------------------ bench */

static int cmd_bench(int argc, char **argv) {
    uint32_t positions[16] = { 8192u, 32768u, 62234u, 100000u };
    uint32_t n_positions = 4;
    uint32_t tokens = 40, reps = 5;
    int use_env = 0;   /* skip the A/B override and honour the kill switches */

    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--pos") && i + 1 < argc) {
            n_positions = 0;
            char *s = strdup(argv[++i]), *tok = strtok(s, ",");
            while (tok && n_positions < 16) { positions[n_positions++] = (uint32_t)strtoul(tok, NULL, 10); tok = strtok(NULL, ","); }
            free(s);
        } else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--env")) use_env = 1;
    }

    printf("%-7s %-9s %-8s %-7s %10s %10s %10s %10s\n",
           "path", "pos", "n_comp", "n_pools", "score/tok", "topk/tok", "expand/tk", "chain/tok");
    for (uint32_t p = 0; p < n_positions; p++) {
        chain_t c;
        if (!chain_alloc(&c, positions[p])) return 1;
        for (int fast = use_env ? 2 : 0; fast <= 2; fast++) {
            if (fast < 2) ds4_gpu_glm_indexer_select_override(fast, fast);
            else          ds4_gpu_glm_indexer_select_override(-1, -1);
            double s = bench_stages(&c, ST_SCORE, tokens, reps);
            double k = bench_stages(&c, ST_TOPK, tokens, reps);
            double e = bench_stages(&c, ST_EXPAND, tokens, reps);
            double a = bench_stages(&c, ST_ALL, tokens, reps);
            printf("%-7s %-9u %-8u %-7u %10.1f %10.1f %10.1f %10.1f\n",
                   fast == 2 ? "env" : (fast ? "fast" : "legacy"),
                   positions[p], c.n_comp, c.selected_pools, s, k, e, a);
            if (use_env) break;
            fflush(stdout);
        }
        ds4_gpu_glm_indexer_select_override(-1, -1);
        chain_free(&c);
    }
    return 0;
}

/* --------------------------------------------------------------- identity */

static void fill_scores_nan(float *v, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t r = rng_u32() % 5u;
        v[i] = r == 0 ? DS4_BENCH_NAN : rng_f32_unit();
    }
}

static int cmd_identity_topk(int argc, char **argv) {
    uint32_t cases = 10000, n_comp_list[8] = { 15558u, 2048u, 8192u, 25000u }, n_shapes = 4;
    uint32_t nan_cases = 2000;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--cases") && i + 1 < argc) cases = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--nan-cases") && i + 1 < argc) nan_cases = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--n-comp") && i + 1 < argc) {
            n_shapes = 0;
            char *sp = strdup(argv[++i]), *tok = strtok(sp, ",");
            while (tok && n_shapes < 8) { n_comp_list[n_shapes++] = (uint32_t)strtoul(tok, NULL, 10); tok = strtok(NULL, ","); }
            free(sp);
        }
    }

    uint32_t max_n = 0;
    for (uint32_t i = 0; i < n_shapes; i++) if (n_comp_list[i] > max_n) max_n = n_comp_list[i];

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc((uint64_t)max_n * sizeof(float));
    ds4_gpu_tensor *sel_a  = ds4_gpu_tensor_alloc((uint64_t)max_n * sizeof(uint32_t));
    ds4_gpu_tensor *sel_b  = ds4_gpu_tensor_alloc((uint64_t)max_n * sizeof(uint32_t));
    float    *h_scores = malloc((size_t)max_n * sizeof(float));
    uint32_t *h_a = malloc((size_t)max_n * sizeof(uint32_t));
    uint32_t *h_b = malloc((size_t)max_n * sizeof(uint32_t));
    if (!scores || !sel_a || !sel_b || !h_scores || !h_a || !h_b) return 1;

    uint64_t finite_checked = 0, finite_bad = 0, nan_checked = 0, nan_bad = 0;
    uint32_t first_bad_case = 0;

    for (uint32_t ci = 0; ci < cases + nan_cases; ci++) {
        const int nan_case = ci >= cases;
        uint32_t n_comp = n_comp_list[ci % n_shapes];
        /* mix of top_k values around the production 512 */
        static const uint32_t topks[] = { 512u, 1u, 2u, 511u, 513u, 1024u, 64u, 2048u };
        uint32_t top_k = topks[(ci / n_shapes) % 8u];
        if (top_k > n_comp) top_k = n_comp;

        if (nan_case) fill_scores_nan(h_scores, n_comp);
        else          fill_scores(h_scores, n_comp, ci);
        ds4_gpu_tensor_write(scores, 0, h_scores, (uint64_t)n_comp * sizeof(float));

        ds4_gpu_glm_indexer_select_override(-1, 0);   /* legacy chain */
        if (!ds4_gpu_indexer_topk_tensor(sel_a, scores, n_comp, 1, top_k)) return 1;
        ds4_gpu_tensor_read(sel_a, 0, h_a, (uint64_t)top_k * sizeof(uint32_t));

        ds4_gpu_glm_indexer_select_override(-1, 1);   /* fast chain */
        if (!ds4_gpu_indexer_topk_tensor(sel_b, scores, n_comp, 1, top_k)) return 1;
        ds4_gpu_tensor_read(sel_b, 0, h_b, (uint64_t)top_k * sizeof(uint32_t));

        int bad = memcmp(h_a, h_b, (size_t)top_k * sizeof(uint32_t)) != 0;
        if (nan_case) { nan_checked++; nan_bad += (uint64_t)bad; }
        else          { finite_checked++; finite_bad += (uint64_t)bad; }
        if (bad && !first_bad_case) {
            first_bad_case = ci + 1;
            uint32_t j = 0;
            while (j < top_k && h_a[j] == h_b[j]) j++;
            fprintf(stderr, "MISMATCH case=%u n_comp=%u top_k=%u nan=%d at slot %u: old=%u new=%u "
                            "(score old=%.9g new=%.9g)\n",
                    ci, n_comp, top_k, nan_case, j, h_a[j], h_b[j],
                    (double)h_scores[h_a[j]], (double)h_scores[h_b[j]]);
        }
    }
    ds4_gpu_glm_indexer_select_override(-1, -1);

    printf("identity-topk: finite cases=%llu mismatches=%llu | nan cases=%llu mismatches=%llu\n",
           (unsigned long long)finite_checked, (unsigned long long)finite_bad,
           (unsigned long long)nan_checked, (unsigned long long)nan_bad);
    free(h_scores); free(h_a); free(h_b);
    ds4_gpu_tensor_free(scores); ds4_gpu_tensor_free(sel_a); ds4_gpu_tensor_free(sel_b);
    return finite_bad != 0;
}

static int cmd_identity_score(int argc, char **argv) {
    uint32_t cases = 50000, max_n = 4096;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--cases") && i + 1 < argc) cases = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--max-n") && i + 1 < argc) max_n = (uint32_t)strtoul(argv[++i], NULL, 10);
    }

    ds4_gpu_tensor *q   = ds4_gpu_tensor_alloc((uint64_t)N_INDEXER_HEAD * N_INDEXER_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *w   = ds4_gpu_tensor_alloc((uint64_t)N_INDEXER_HEAD * sizeof(float));
    ds4_gpu_tensor *k16 = ds4_gpu_tensor_alloc((uint64_t)max_n * N_INDEXER_HEAD_DIM * sizeof(uint16_t));
    ds4_gpu_tensor *k32 = ds4_gpu_tensor_alloc((uint64_t)max_n * N_INDEXER_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *sa  = ds4_gpu_tensor_alloc((uint64_t)max_n * sizeof(float));
    ds4_gpu_tensor *sb  = ds4_gpu_tensor_alloc((uint64_t)max_n * sizeof(float));
    float *hq = malloc((size_t)N_INDEXER_HEAD * N_INDEXER_HEAD_DIM * sizeof(float));
    float *hw = malloc((size_t)N_INDEXER_HEAD * sizeof(float));
    uint16_t *hk16 = malloc((size_t)max_n * N_INDEXER_HEAD_DIM * sizeof(uint16_t));
    float    *hk32 = malloc((size_t)max_n * N_INDEXER_HEAD_DIM * sizeof(float));
    float *ha = malloc((size_t)max_n * sizeof(float));
    float *hb = malloc((size_t)max_n * sizeof(float));
    if (!q || !w || !k16 || !k32 || !sa || !sb || !hq || !hw || !hk16 || !hk32 || !ha || !hb) return 1;

    uint64_t rows_checked = 0, bad_rows = 0, bad_cases = 0;
    for (uint32_t ci = 0; ci < cases; ci++) {
        const uint32_t n = 1u + (rng_u32() % max_n);
        const int f16 = (ci & 1u) == 0u;
        const float amp = (ci % 5u == 0u) ? 64.0f : 1.0f;
        for (uint32_t i = 0; i < N_INDEXER_HEAD * N_INDEXER_HEAD_DIM; i++)
            hq[i] = (rng_f32_unit() - 0.5f) * 2.0f * amp;
        for (uint32_t i = 0; i < N_INDEXER_HEAD; i++)
            hw[i] = (ci % 3u == 0u) ? (rng_f32_unit() - 0.5f) * 4.0f : rng_f32_unit();
        for (uint64_t i = 0; i < (uint64_t)n * N_INDEXER_HEAD_DIM; i++) {
            float v = (rng_f32_unit() - 0.5f) * 2.0f * amp;
            hk32[i] = v;
            hk16[i] = f32_to_f16(v);
        }
        ds4_gpu_tensor_write(q, 0, hq, (uint64_t)N_INDEXER_HEAD * N_INDEXER_HEAD_DIM * sizeof(float));
        ds4_gpu_tensor_write(w, 0, hw, (uint64_t)N_INDEXER_HEAD * sizeof(float));
        if (f16) ds4_gpu_tensor_write(k16, 0, hk16, (uint64_t)n * N_INDEXER_HEAD_DIM * sizeof(uint16_t));
        else     ds4_gpu_tensor_write(k32, 0, hk32, (uint64_t)n * N_INDEXER_HEAD_DIM * sizeof(float));
        ds4_gpu_tensor *keys = f16 ? k16 : k32;

        ds4_gpu_glm_indexer_select_override(0, -1);
        if (!ds4_gpu_glm_indexer_score_one_tensor(sa, q, w, keys, n, N_INDEXER_HEAD,
                                                  N_INDEXER_HEAD_DIM, indexer_scale(), f16)) return 1;
        ds4_gpu_tensor_read(sa, 0, ha, (uint64_t)n * sizeof(float));

        ds4_gpu_glm_indexer_select_override(1, -1);
        if (!ds4_gpu_glm_indexer_score_one_tensor(sb, q, w, keys, n, N_INDEXER_HEAD,
                                                  N_INDEXER_HEAD_DIM, indexer_scale(), f16)) return 1;
        ds4_gpu_tensor_read(sb, 0, hb, (uint64_t)n * sizeof(float));

        uint64_t case_bad = 0;
        for (uint32_t r = 0; r < n; r++) {
            uint32_t ua, ub;
            memcpy(&ua, &ha[r], 4); memcpy(&ub, &hb[r], 4);
            if (ua != ub) {
                if (!bad_rows) fprintf(stderr, "SCORE MISMATCH case=%u row=%u f16=%d old=%.9g(%08x) new=%.9g(%08x)\n",
                                       ci, r, f16, (double)ha[r], ua, (double)hb[r], ub);
                case_bad++;
            }
        }
        rows_checked += n;
        bad_rows += case_bad;
        bad_cases += (case_bad != 0);
    }
    ds4_gpu_glm_indexer_select_override(-1, -1);
    printf("identity-score: cases=%u rows=%llu differing_rows=%llu differing_cases=%llu\n",
           cases, (unsigned long long)rows_checked, (unsigned long long)bad_rows,
           (unsigned long long)bad_cases);
    return bad_rows != 0;
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }
    const char *cmd = argc > 1 ? argv[1] : "bench";
    if (!strcmp(cmd, "bench"))           return cmd_bench(argc - 2, argv + 2);
    if (!strcmp(cmd, "identity-topk"))   return cmd_identity_topk(argc - 2, argv + 2);
    if (!strcmp(cmd, "identity-score"))  return cmd_identity_score(argc - 2, argv + 2);
    if (!strcmp(cmd, "identity-expand")) return cmd_identity_expand(argc - 2, argv + 2);
    fprintf(stderr, "unknown command %s\n", cmd);
    return 1;
}
