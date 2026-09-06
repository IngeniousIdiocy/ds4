/* Stage-1 test vehicle for the GLM-5.3 expanded-expert one-layer bank.
 *
 * Drives the PRODUCTION routed-MoE entry point ds4_gpu_glm_routed_moe_batch_tensor
 * on one real routed layer of the public GGUF, twice on identical operands:
 *
 *   arm PACKED  the shipped chain (bank disarmed)
 *   arm BANK    the same chain with this layer's bank armed
 *
 * and reports
 *   1. bank identity  -- every half of the three bank sections re-derived from
 *      the packed rows through the inverse index mapping (--verify),
 *   2. output identity -- every output word of the captured prompt prefix,
 *      the nine production chunks of the packed arm against the one
 *      whole-prompt batch of the bank arm, with planted single-row controls so
 *      the comparator is proven live,
 *   3. complete-routed-segment timing (map + gate + up + SwiGLU + down + sum8,
 *      expansion charged once per layer) at the production chunk boundaries
 *      and at the whole-prompt batch, in the same x42 per-token convention as
 *      the Stage-0 harness receipt.
 *
 * Routing is the real capture from a prefill of the reference prompt on the
 * public file (DS4_GLM_DUMP_MOE_IDS).  Activations are deterministic and
 * identical between the arms, which is what the bit-exactness claim rests on.
 *
 * GPU test: the caller runs it under /tmp/ds4-gpu-lock with the model process
 * down.  Peak Metal residency at 62,174 tokens is about 39 GiB.
 */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
    DIN = 4096,          /* expert_in_dim  / out_dim */
    DMID = 2048,         /* expert_mid_dim */
    TOPK = 8,
    NEXP = 288,
    Q4_K = 12,           /* DS4_METAL_TENSOR_Q4_K */
    TRUNK_MOE_LAYERS = 42 /* the x42 convention of the design and the receipt */
};
#define SWIGLU_CLAMP 10.0f

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* mincore observes VM residency without touching the mapped pages.  It does
 * not prove that pages are wired, and the OS may evict them after the sample;
 * the snapshots only explain which costs surrounded each timed expansion. */
static int report_model_residency(const void *map,
                                  uint64_t map_bytes,
                                  uint64_t span_off,
                                  uint64_t span_len,
                                  const char *label) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t lo = span_off & ~(page - 1u);
    const uint64_t hi0 = span_off + span_len;
    const uint64_t hi = (hi0 + page - 1u) & ~(page - 1u);
    if (hi0 < span_off || hi > map_bytes || hi <= lo) return 0;
    const uint64_t pages = (hi - lo) / page;
    if (pages > SIZE_MAX) return 0;
    unsigned char *vec = calloc((size_t)pages, 1u);
    if (!vec) return 0;
    if (mincore((char *)map + lo, (size_t)(hi - lo), (char *)vec) != 0) {
        perror("mincore");
        free(vec);
        return 0;
    }
    uint64_t resident = 0;
    for (uint64_t i = 0; i < pages; i++) resident += (vec[i] & 1u) != 0;
    printf("model pages %-20s: %llu/%llu resident (%.2f%%, %.3f/%.3f GiB)\n",
           label,
           (unsigned long long)resident,
           (unsigned long long)pages,
           pages ? 100.0 * (double)resident / (double)pages : 0.0,
           (double)(resident * page) / 1073741824.0,
           (double)(pages * page) / 1073741824.0);
    free(vec);
    return 1;
}

/* ---------------- real routing capture ---------------- */
static int32_t *g_ids;
static uint32_t g_total_tokens;
static uint32_t g_prod_chunks[512];
static int      g_nprod;

static void load_ids(const char *path,
                     uint32_t    want_layer,
                     uint32_t    token_limit) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open ids dump %s\n", path); exit(2); }
    size_t cap = 1u << 20, used = 0;
    g_ids = malloc(cap * sizeof(int32_t));
    uint32_t hdr[3];
    while (g_ids && fread(hdr, sizeof(uint32_t), 3, f) == 3) {
        const uint32_t layer = hdr[0], nt = hdr[1], ne = hdr[2];
        const size_t n = (size_t)nt * ne;
        int32_t *buf = malloc(n * sizeof(int32_t));
        if (!buf || fread(buf, sizeof(int32_t), n, f) != n) { free(buf); break; }
        if (ne != TOPK || layer != want_layer) { free(buf); continue; }
        const uint32_t have_tokens = (uint32_t)(used / TOPK);
        uint32_t take = nt;
        if (token_limit != 0) {
            if (have_tokens >= token_limit) take = 0;
            else if (take > token_limit - have_tokens) {
                take = token_limit - have_tokens;
            }
        }
        const size_t take_ids = (size_t)take * ne;
        while (used + take_ids > cap) {
            cap *= 2;
            g_ids = realloc(g_ids, cap * sizeof(int32_t));
        }
        if (!g_ids) { free(buf); break; }
        memcpy(g_ids + used, buf, take_ids * sizeof(int32_t));
        used += take_ids;
        free(buf);
        if (take != 0 && g_nprod < 512) g_prod_chunks[g_nprod++] = take;
    }
    fclose(f);
    if (!g_ids) { fprintf(stderr, "out of memory loading routing\n"); exit(2); }
    g_total_tokens = (uint32_t)(used / TOPK);
    if (token_limit != 0 && g_total_tokens != token_limit) {
        fprintf(stderr,
                "requested %u routing tokens for layer %u, capture has %u\n",
                token_limit, want_layer, g_total_tokens);
        exit(2);
    }
    printf("routing: layer %u, %u tokens, %d production chunks (",
           want_layer, g_total_tokens, g_nprod);
    for (int i = 0; i < g_nprod; i++) printf("%s%u", i ? ", " : "", g_prod_chunks[i]);
    printf(")\n");
    if (g_total_tokens == 0) {
        fprintf(stderr, "no routing for layer %u in %s\n", want_layer, path);
        exit(2);
    }
}

/* ---------------- deterministic operands ---------------- */
static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16; return x;
}
static float unit(uint32_t h) {
    return ((float)(h >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

/* ---------------- CPU self-test: are the requested ranges mappable? ----------
 * The bank expansion and the packed control both read the three expert tensors
 * through the Metal model views.  A view set that does not COVER a requested
 * [offset, offset+bytes) range makes ds4_gpu_wrap_model_range() refuse, which
 * is how the first run failed ("Metal model range 84.85..86.12 GiB is not
 * covered by mapped model views"): the vehicle mmapped the file but never
 * registered it.  This check runs before ds4_gpu_init(), so a bad offset set
 * costs no GPU time at all. */
static int check_ranges(uint64_t file_bytes,
                        const uint64_t *offs,
                        const uint64_t *lens,
                        int n,
                        uint64_t *span_off,
                        uint64_t *span_len,
                        uint64_t *max_tensor) {
    int ok = 1;
    uint64_t lo = UINT64_MAX, hi = 0, mx = 0;
    for (int i = 0; i < n; i++) {
        printf("  range %d: %llu .. %llu (%.3f GiB)", i,
               (unsigned long long)offs[i],
               (unsigned long long)(offs[i] + lens[i]),
               lens[i] / 1073741824.0);
        if (lens[i] == 0 || offs[i] > file_bytes || lens[i] > file_bytes - offs[i]) {
            printf("  OUTSIDE THE FILE (%llu bytes)\n",
                   (unsigned long long)file_bytes);
            ok = 0;
            continue;
        }
        printf("  ok\n");
        if (offs[i] < lo) lo = offs[i];
        if (offs[i] + lens[i] > hi) hi = offs[i] + lens[i];
        if (lens[i] > mx) mx = lens[i];
    }
    if (!ok) return 0;
    *span_off = lo;
    *span_len = hi - lo;
    *max_tensor = mx;
    printf("  span to map: %llu .. %llu (%.3f GiB), largest tensor %.3f GiB\n",
           (unsigned long long)lo, (unsigned long long)hi,
           (hi - lo) / 1073741824.0, mx / 1073741824.0);
    return 1;
}

/* ---------------- one routed-segment call ---------------- */
typedef struct {
    const void *map;
    uint64_t    map_size;
    uint64_t    gate_off, up_off, down_off;
    uint32_t    layer;
    ds4_gpu_tensor *x, *sel, *wts, *mid, *out;
} bank_ctx;

static int routed_call(const bank_ctx *c, uint32_t t0, uint32_t nt) {
    const uint64_t gate_row = (uint64_t)DIN / 256u * 144u;   /* 2304 */
    const uint64_t down_row = (uint64_t)DMID / 256u * 144u;  /* 1152 */
    ds4_gpu_tensor *xv = ds4_gpu_tensor_view(c->x, (uint64_t)t0 * DIN * sizeof(float),
                                             (uint64_t)nt * DIN * sizeof(float));
    ds4_gpu_tensor *sv = ds4_gpu_tensor_view(c->sel, (uint64_t)t0 * TOPK * sizeof(int32_t),
                                             (uint64_t)nt * TOPK * sizeof(int32_t));
    ds4_gpu_tensor *wv = ds4_gpu_tensor_view(c->wts, (uint64_t)t0 * TOPK * sizeof(float),
                                             (uint64_t)nt * TOPK * sizeof(float));
    ds4_gpu_tensor *ov = ds4_gpu_tensor_view(c->out, (uint64_t)t0 * DIN * sizeof(float),
                                             (uint64_t)nt * DIN * sizeof(float));
    int ok = xv && sv && wv && ov;
    if (ok) {
        ok = ds4_gpu_glm_routed_moe_batch_tensor(
                ov, c->mid, c->map, c->map_size,
                c->gate_off, c->up_off, c->down_off,
                Q4_K, Q4_K, Q4_K,
                (uint64_t)DMID * gate_row, gate_row,
                (uint64_t)DMID * gate_row, gate_row,
                (uint64_t)DIN * down_row, down_row,
                DIN, DMID, DIN,
                sv, wv, NEXP, TOPK, SWIGLU_CLAMP, c->layer,
                xv, nt, TOPK * DMID, true);
    }
    ds4_gpu_tensor_free(ov);
    ds4_gpu_tensor_free(wv);
    ds4_gpu_tensor_free(sv);
    ds4_gpu_tensor_free(xv);
    return ok;
}

/* One complete pass over the whole prompt at the given schedule. */
static int pass(const bank_ctx *c, int whole) {
    if (whole) return routed_call(c, 0, g_total_tokens);
    uint32_t t0 = 0;
    for (int i = 0; i < g_nprod; i++) {
        if (!routed_call(c, t0, g_prod_chunks[i])) return 0;
        t0 += g_prod_chunks[i];
    }
    return 1;
}

static void stats(const double *v, int n, double *mean, double *lo, double *hi) {
    *mean = 0; *lo = v[0]; *hi = v[0];
    for (int i = 0; i < n; i++) {
        *mean += v[i];
        if (v[i] < *lo) *lo = v[i];
        if (v[i] > *hi) *hi = v[i];
    }
    *mean /= n;
}

static int observe_expansion_pair(const void *map,
                                  uint64_t map_bytes,
                                  uint64_t span_off,
                                  uint64_t span_len,
                                  uint64_t gate_off,
                                  uint64_t up_off,
                                  uint64_t down_off,
                                  uint32_t layer,
                                  uint32_t tokens,
                                  double *cold_ms_out,
                                  double *warm_ms_out) {
    const uint64_t gate_row = (uint64_t)DIN / 256u * 144u;
    const uint64_t down_row = (uint64_t)DMID / 256u * 144u;
    const uint64_t bank_bytes =
        ds4_gpu_glm_expert_bank_bytes(DIN, DMID, DIN, NEXP);
    ds4_gpu_glm_expert_bank_stats before, cold, warm;
    ds4_gpu_glm_expert_bank_get_stats(&before);

    const double cold_t0 = now_ms();
    int armed = ds4_gpu_glm_expert_bank_expand_layer(
            map, map_bytes, gate_off, up_off, down_off,
            Q4_K, Q4_K, Q4_K,
            (uint64_t)DMID * gate_row, gate_row,
            (uint64_t)DMID * gate_row, gate_row,
            (uint64_t)DIN * down_row, down_row,
            DIN, DMID, DIN, NEXP, layer);
    const double cold_ms = now_ms() - cold_t0;
    if (!armed) return 0;
    ds4_gpu_glm_expert_bank_get_stats(&cold);
    (void)report_model_residency(map, map_bytes, span_off, span_len,
                                 "after first expand");

    const double warm_t0 = now_ms();
    armed = ds4_gpu_glm_expert_bank_expand_layer(
            map, map_bytes, gate_off, up_off, down_off,
            Q4_K, Q4_K, Q4_K,
            (uint64_t)DMID * gate_row, gate_row,
            (uint64_t)DMID * gate_row, gate_row,
            (uint64_t)DIN * down_row, down_row,
            DIN, DMID, DIN, NEXP, layer);
    const double warm_ms = now_ms() - warm_t0;
    if (!armed) return 0;
    ds4_gpu_glm_expert_bank_get_stats(&warm);
    (void)report_model_residency(map, map_bytes, span_off, span_len,
                                 "after warm expand");

    printf("expansion first observed: %.3f ms = %.2f us/token x%d "
           "(ensure %.3f, model-view %.3f, command %.3f ms)\n",
           cold_ms, cold_ms * 1000.0 * TRUNK_MOE_LAYERS / tokens,
           TRUNK_MOE_LAYERS, cold.last_ensure_ms,
           cold.last_model_view_ms, cold.last_command_ms);
    printf("expansion same-layer warm: %.3f ms = %.2f us/token x%d "
           "(ensure %.3f, model-view %.3f, command %.3f ms; resident-kernel floor)\n",
           warm_ms, warm_ms * 1000.0 * TRUNK_MOE_LAYERS / tokens,
           TRUNK_MOE_LAYERS, warm.last_ensure_ms,
           warm.last_model_view_ms, warm.last_command_ms);
    printf("bank allocation reuse: allocations %llu -> %llu -> %llu, "
           "expansions %llu -> %llu -> %llu, capacity %.3f GiB, "
           "Metal allocated %.3f -> %.3f -> %.3f GiB\n",
           (unsigned long long)before.allocation_count,
           (unsigned long long)cold.allocation_count,
           (unsigned long long)warm.allocation_count,
           (unsigned long long)before.expansion_count,
           (unsigned long long)cold.expansion_count,
           (unsigned long long)warm.expansion_count,
           warm.capacity_bytes / 1073741824.0,
           before.current_allocated_bytes / 1073741824.0,
           cold.current_allocated_bytes / 1073741824.0,
           warm.current_allocated_bytes / 1073741824.0);

    *cold_ms_out = cold_ms;
    *warm_ms_out = warm_ms;
    return cold.allocation_count == before.allocation_count + 1u &&
           warm.allocation_count == cold.allocation_count &&
           cold.expansion_count == before.expansion_count + 1u &&
           warm.expansion_count == cold.expansion_count + 1u &&
           warm.capacity_bytes >= bank_bytes;
}

int main(int argc, char **argv) {
    const char *gguf = NULL, *ids = NULL;
    uint32_t layer = 24;
    uint32_t token_limit = 0;
    uint64_t gate_off = 0, up_off = 0, down_off = 0;
    int passes = 8, do_verify = 0, do_compare = 1, prod = 1, whole = 1;
    int self_test_only = 0, expansion_only = 0, cross_screen = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--gguf") && v) { gguf = v; i++; }
        else if (!strcmp(a, "--ids") && v) { ids = v; i++; }
        else if (!strcmp(a, "--layer") && v) { layer = (uint32_t)strtoul(v, NULL, 10); i++; }
        else if (!strcmp(a, "--tokens") && v) { token_limit = (uint32_t)strtoul(v, NULL, 10); i++; }
        else if (!strcmp(a, "--off-gate") && v) { gate_off = strtoull(v, NULL, 10); i++; }
        else if (!strcmp(a, "--off-up") && v) { up_off = strtoull(v, NULL, 10); i++; }
        else if (!strcmp(a, "--off-down") && v) { down_off = strtoull(v, NULL, 10); i++; }
        else if (!strcmp(a, "--passes") && v) { passes = atoi(v); i++; }
        else if (!strcmp(a, "--verify")) do_verify = 1;
        else if (!strcmp(a, "--self-test")) self_test_only = 1;
        else if (!strcmp(a, "--expansion-only")) expansion_only = 1;
        else if (!strcmp(a, "--cross-screen")) cross_screen = 1;
        else if (!strcmp(a, "--no-compare")) do_compare = 0;
        else if (!strcmp(a, "--chunks") && v) {
            prod = strstr(v, "prod") != NULL;
            whole = strstr(v, "whole") != NULL;
            i++;
        } else {
            fprintf(stderr,
                    "usage: %s --gguf F --ids F --layer N --off-gate N --off-up N "
                    "--off-down N [--tokens N] [--chunks prod,whole] [--passes N] "
                    "[--verify] [--no-compare] [--self-test] [--expansion-only] "
                    "[--cross-screen]\n", argv[0]);
            return 2;
        }
    }
    if (!gguf || !ids || !gate_off || !up_off || !down_off) {
        fprintf(stderr, "missing --gguf / --ids / --off-*\n");
        return 2;
    }
    if (passes < 1) passes = 1;

    /* The bank branch is opt-in; arming and disarming selects the arm. */
    setenv("DS4_GLM_ENABLE_EXPERT_BANK", "1", 1);
    unsetenv("DS4_GLM_DISABLE_EXPERT_BANK");

    load_ids(ids, layer, token_limit);

    int fd = open(gguf, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) { perror("open gguf"); return 2; }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap gguf"); return 2; }

    /* CPU self-test first: no Metal call has happened yet. */
    const uint64_t gate_row_b = (uint64_t)DIN / 256u * 144u;   /* 2304 */
    const uint64_t down_row_b = (uint64_t)DMID / 256u * 144u;  /* 1152 */
    const uint64_t offs[3] = { gate_off, up_off, down_off };
    const uint64_t lens[3] = { (uint64_t)NEXP * DMID * gate_row_b,
                               (uint64_t)NEXP * DMID * gate_row_b,
                               (uint64_t)NEXP * DIN * down_row_b };
    uint64_t span_off = 0, span_len = 0, max_tensor = 0;
    printf("expert tensor ranges for layer %u:\n", layer);
    if (!check_ranges((uint64_t)st.st_size, offs, lens, 3,
                      &span_off, &span_len, &max_tensor)) {
        fprintf(stderr,
                "FAIL: the requested expert offsets are not inside the GGUF; "
                "no GPU work attempted\n");
        return 2;
    }
    if (self_test_only) {
        printf("test_glm_expert_bank: --self-test OK (offsets checked, no GPU "
               "work)\n");
        return 0;
    }

    (void)report_model_residency(map, (uint64_t)st.st_size,
                                 span_off, span_len, "after mmap");

    if (!ds4_gpu_init()) { fprintf(stderr, "Metal init failed\n"); return 2; }
    /* Register the span with Metal.  ds4_gpu_wrap_model_range() resolves the
     * absolute offsets against these views; without this the expansion and the
     * packed control both refuse.  max_tensor_bytes is the largest single
     * expert tensor so every one of them lands inside ONE view. */
    if (!ds4_gpu_set_model_map_range(map, (uint64_t)st.st_size,
                                     span_off, span_len, max_tensor)) {
        fprintf(stderr, "FAIL: could not register the model span with Metal\n");
        return 2;
    }
    printf("registered %.3f GiB of model views from offset %llu\n",
           span_len / 1073741824.0, (unsigned long long)span_off);
    (void)report_model_residency(map, (uint64_t)st.st_size,
                                 span_off, span_len, "after Metal mapping");

    const uint64_t bank_bytes = ds4_gpu_glm_expert_bank_bytes(DIN, DMID, DIN, NEXP);
    printf("bank: %.3f GiB for layer %u (%u experts)\n",
           bank_bytes / 1073741824.0, layer, (unsigned)NEXP);
    double cold_expand_ms = 0.0, warm_expand_ms = 0.0;
    int failures = observe_expansion_pair(
            map, (uint64_t)st.st_size, span_off, span_len,
            gate_off, up_off, down_off, layer, g_total_tokens,
            &cold_expand_ms, &warm_expand_ms) ? 0 : 1;
    if (failures) {
        fprintf(stderr, "FAIL: bank expansion or allocation reuse failed\n");
        return 1;
    }
    if (expansion_only) {
        printf("test_glm_expert_bank: expansion-only OK\n");
        ds4_gpu_glm_expert_bank_free();
        return 0;
    }

    const uint32_t T = g_total_tokens;
    const uint64_t pairs = (uint64_t)T * TOPK;
    bank_ctx c = {0};
    c.map = map; c.map_size = (uint64_t)st.st_size; c.layer = layer;
    c.gate_off = gate_off; c.up_off = up_off; c.down_off = down_off;
    c.x = ds4_gpu_tensor_alloc((uint64_t)T * DIN * sizeof(float));
    c.sel = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
    c.wts = ds4_gpu_tensor_alloc(pairs * sizeof(float));
    c.mid = ds4_gpu_tensor_alloc(pairs * DMID * sizeof(uint16_t));
    c.out = ds4_gpu_tensor_alloc((uint64_t)T * DIN * sizeof(float));
    if (!c.x || !c.sel || !c.wts || !c.mid || !c.out) {
        fprintf(stderr, "tensor allocation failed\n");
        return 2;
    }

    /* Deterministic activations and routing weights; the two arms see exactly
     * these bytes, so any output difference is the bank's doing. */
    float *xh = malloc((uint64_t)T * DIN * sizeof(float));
    float *wh = malloc(pairs * sizeof(float));
    if (!xh || !wh) { fprintf(stderr, "host allocation failed\n"); return 2; }
    for (uint64_t i = 0; i < (uint64_t)T * DIN; i++) xh[i] = unit(mix32((uint32_t)i)) * 0.5f;
    for (uint64_t i = 0; i < pairs; i++) wh[i] = 0.125f + unit(mix32((uint32_t)i + 7u)) * 0.01f;
    if (!ds4_gpu_tensor_write(c.x, 0, xh, (uint64_t)T * DIN * sizeof(float)) ||
        !ds4_gpu_tensor_write(c.wts, 0, wh, pairs * sizeof(float)) ||
        !ds4_gpu_tensor_write(c.sel, 0, g_ids, pairs * sizeof(int32_t))) {
        fprintf(stderr, "operand upload failed\n");
        return 2;
    }

    const uint64_t gate_row = gate_row_b;
    const uint64_t down_row = down_row_b;
    /* ---- 1. bank identity ---- */
    if (do_verify) {
        static const char *names[3] = { "gate", "up", "down" };
        const uint64_t offs[3] = { gate_off, up_off, down_off };
        const uint64_t ebytes[3] = { (uint64_t)DMID * gate_row,
                                     (uint64_t)DMID * gate_row,
                                     (uint64_t)DIN * down_row };
        const uint64_t rbytes[3] = { gate_row, gate_row, down_row };
        const uint32_t rows[3] = { DMID, DMID, DIN };
        const uint32_t ne00[3] = { DIN, DIN, DMID };
        uint64_t total_cmp = 0, total_mism = 0, total_fin = 0;
        for (int s = 0; s < 3; s++) {
            uint64_t ctr[4] = {0};
            if (!ds4_gpu_glm_expert_bank_verify_section(
                    map, (uint64_t)st.st_size, offs[s], ebytes[s], rbytes[s],
                    rows[s], ne00[s], NEXP, s, ctr)) {
                fprintf(stderr, "FAIL: bank verify dispatch failed for %s\n", names[s]);
                failures++;
                continue;
            }
            printf("bank verify %-4s: compared %llu, mismatching %llu, finite %llu\n",
                   names[s], (unsigned long long)ctr[1],
                   (unsigned long long)ctr[0], (unsigned long long)ctr[3]);
            total_cmp += ctr[1]; total_mism += ctr[0]; total_fin += ctr[3];
            if (ctr[0] != 0) failures++;
        }
        printf("bank verify TOTAL: %llu halves compared, %llu mismatching, %llu finite\n",
               (unsigned long long)total_cmp, (unsigned long long)total_mism,
               (unsigned long long)total_fin);
    }

    /* ---- 2. output identity: 9 packed chunks vs 1 bank batch ---- */
    float *ref = NULL, *got = NULL;
    if (do_compare) {
        ref = malloc((uint64_t)T * DIN * sizeof(float));
        got = malloc((uint64_t)T * DIN * sizeof(float));
        if (!ref || !got) { fprintf(stderr, "compare buffers failed\n"); return 2; }

        for (int planted = 0; planted <= 1; planted++) {
            if (planted) {
                /* Control: one activation word changed, so a live comparator
                 * must report mismatches.  Restored afterwards. */
                float save = xh[(uint64_t)(T / 3) * DIN + 11];
                xh[(uint64_t)(T / 3) * DIN + 11] = save + 1.0f;
                if (!ds4_gpu_tensor_write(c.x, 0, xh,
                                          (uint64_t)T * DIN * sizeof(float))) {
                    fprintf(stderr, "control upload failed\n"); return 2;
                }
                xh[(uint64_t)(T / 3) * DIN + 11] = save;
            }
            ds4_gpu_glm_expert_bank_disarm();
            if (!ds4_gpu_tensor_fill_f32(c.out, NAN, (uint64_t)T * DIN) ||
                !pass(&c, 0) ||
                !ds4_gpu_tensor_read(c.out, 0, ref, (uint64_t)T * DIN * sizeof(float))) {
                fprintf(stderr, "FAIL: packed arm failed\n"); return 1;
            }
            if (planted) {
                /* the perturbed input goes only through the packed arm; put
                 * the pristine operands back for the bank arm so the two
                 * differ by construction */
                if (!ds4_gpu_tensor_write(c.x, 0, xh,
                                          (uint64_t)T * DIN * sizeof(float))) {
                    fprintf(stderr, "control restore failed\n"); return 2;
                }
            }
            if (!ds4_gpu_glm_expert_bank_expand_layer(
                    map, (uint64_t)st.st_size, gate_off, up_off, down_off,
                    Q4_K, Q4_K, Q4_K,
                    (uint64_t)DMID * gate_row, gate_row,
                    (uint64_t)DMID * gate_row, gate_row,
                    (uint64_t)DIN * down_row, down_row,
                    DIN, DMID, DIN, NEXP, layer)) {
                fprintf(stderr, "FAIL: re-arm failed\n"); return 1;
            }
            if (!ds4_gpu_tensor_fill_f32(c.out, NAN, (uint64_t)T * DIN) ||
                !pass(&c, 1) ||
                !ds4_gpu_tensor_read(c.out, 0, got, (uint64_t)T * DIN * sizeof(float))) {
                fprintf(stderr, "FAIL: bank arm failed\n"); return 1;
            }
            uint64_t diff = 0, nonfinite = 0;
            uint64_t first = UINT64_MAX;
            for (uint64_t i = 0; i < (uint64_t)T * DIN; i++) {
                if (!isfinite(ref[i])) nonfinite++;
                uint32_t a, b;
                memcpy(&a, &ref[i], 4);
                memcpy(&b, &got[i], 4);
                if (a != b) { if (diff == 0) first = i; diff++; }
            }
            printf("%s: %llu output words, %llu differing%s, %llu non-finite in the "
                   "packed arm\n",
                   planted ? "planted control" : "output identity",
                   (unsigned long long)((uint64_t)T * DIN),
                   (unsigned long long)diff,
                   diff ? "" : " (bit-exact)",
                   (unsigned long long)nonfinite);
            if (diff && first != UINT64_MAX) {
                printf("  first differing word %llu: packed %.9g vs bank %.9g\n",
                       (unsigned long long)first, ref[first], got[first]);
            }
            if (!planted && diff != 0) failures++;
            if (planted && diff == 0) {
                fprintf(stderr,
                        "FAIL: planted control produced no difference -- the "
                        "comparator is not live\n");
                failures++;
            }
        }
    }

    /* ---- 3. timing ---- */
    double *pk = malloc((size_t)passes * sizeof(double));
    double *bk = malloc((size_t)passes * sizeof(double));
    if (!pk || !bk) return 2;
    if (cross_screen) {
        ds4_gpu_glm_expert_bank_disarm();
        if (!pass(&c, 0)) {
            fprintf(stderr, "FAIL: packed production-boundary warm-up failed\n");
            return 1;
        }
        for (int p = 0; p < passes; p++) {
            const double t0 = now_ms();
            if (!pass(&c, 0)) {
                fprintf(stderr, "FAIL: packed production-boundary timing failed\n");
                return 1;
            }
            pk[p] = now_ms() - t0;
        }
        if (!ds4_gpu_glm_expert_bank_expand_layer(
                    map, (uint64_t)st.st_size, gate_off, up_off, down_off,
                    Q4_K, Q4_K, Q4_K,
                    (uint64_t)DMID * gate_row, gate_row,
                    (uint64_t)DMID * gate_row, gate_row,
                    (uint64_t)DIN * down_row, down_row,
                    DIN, DMID, DIN, NEXP, layer) ||
            !pass(&c, 1)) {
            fprintf(stderr, "FAIL: bank whole-batch warm-up failed\n");
            return 1;
        }
        for (int p = 0; p < passes; p++) {
            const double t0 = now_ms();
            if (!pass(&c, 1)) {
                fprintf(stderr, "FAIL: bank whole-batch timing failed\n");
                return 1;
            }
            bk[p] = now_ms() - t0;
        }
        double pm, plo, phi, bm, blo, bhi;
        stats(pk, passes, &pm, &plo, &phi);
        stats(bk, passes, &bm, &blo, &bhi);
        const double us_tok =
            (double)TRUNK_MOE_LAYERS * 1000.0 / (double)T;
        printf("\n16k cross-schedule routed screen (%d production chunks -> "
               "one whole batch, layer %u)\n", g_nprod, layer);
        printf("  packed production boundaries: mean %.3f ms "
               "(min %.3f max %.3f spread %.3f) = %.2f us/token x%d\n",
               pm, plo, phi, phi - plo, pm * us_tok, TRUNK_MOE_LAYERS);
        printf("  bank whole batch           : mean %.3f ms "
               "(min %.3f max %.3f spread %.3f) = %.2f us/token x%d\n",
               bm, blo, bhi, bhi - blo, bm * us_tok, TRUNK_MOE_LAYERS);
        printf("  routed saving before expansion: %+.2f us/token x%d\n",
               (pm - bm) * us_tok, TRUNK_MOE_LAYERS);
        printf("  net, first-observed expansion repeated x42: %+.2f us/token\n",
               (pm - bm - cold_expand_ms) * us_tok);
        printf("  net, same-layer-warm expansion repeated x42: %+.2f us/token "
               "(resident lower bound)\n",
               (pm - bm - warm_expand_ms) * us_tok);
    } else for (int sched = 0; sched < 2; sched++) {
        const int is_whole = (sched == 1);
        if ((is_whole && !whole) || (!is_whole && !prod)) continue;
        for (int arm = 0; arm < 2; arm++) {
            double *dst = arm ? bk : pk;
            if (arm) {
                if (!ds4_gpu_glm_expert_bank_expand_layer(
                        map, (uint64_t)st.st_size, gate_off, up_off, down_off,
                        Q4_K, Q4_K, Q4_K,
                        (uint64_t)DMID * gate_row, gate_row,
                        (uint64_t)DMID * gate_row, gate_row,
                        (uint64_t)DIN * down_row, down_row,
                        DIN, DMID, DIN, NEXP, layer)) {
                    fprintf(stderr, "FAIL: re-arm before timing failed\n");
                    return 1;
                }
            } else {
                ds4_gpu_glm_expert_bank_disarm();
            }
            if (!pass(&c, is_whole)) {         /* warm-up */
                fprintf(stderr, "FAIL: timing warm-up failed (arm %d)\n", arm);
                return 1;
            }
            for (int p = 0; p < passes; p++) {
                const double t0 = now_ms();
                if (!pass(&c, is_whole)) {
                    fprintf(stderr, "FAIL: timing pass failed (arm %d)\n", arm);
                    return 1;
                }
                dst[p] = now_ms() - t0;
            }
        }
        double pm, plo, phi, bm, blo, bhi;
        stats(pk, passes, &pm, &plo, &phi);
        stats(bk, passes, &bm, &blo, &bhi);
        const double us_tok = (double)TRUNK_MOE_LAYERS * 1000.0 / (double)T;
        printf("\nschedule %s (%d sub-chunk%s, layer %u)\n",
               is_whole ? "whole prompt" : "production boundaries",
               is_whole ? 1 : g_nprod, (is_whole || g_nprod == 1) ? "" : "s", layer);
        printf("  packed Q4_K : mean %.3f ms (min %.3f max %.3f spread %.3f) "
               "= %.2f us/token x%d\n",
               pm, plo, phi, phi - plo, pm * us_tok, TRUNK_MOE_LAYERS);
        printf("  bank (T)    : mean %.3f ms (min %.3f max %.3f spread %.3f) "
               "= %.2f us/token x%d\n",
               bm, blo, bhi, bhi - blo, bm * us_tok, TRUNK_MOE_LAYERS);
        printf("  net, first-observed expansion repeated x42: %+.2f us/token "
               "(includes one-time allocation; upper charge)\n",
               (pm - bm - cold_expand_ms) * us_tok);
        printf("  net, same-layer-warm expansion repeated x42: %+.2f us/token "
               "(resident lower bound; use native all-layer timing for production)\n",
               (pm - bm - warm_expand_ms) * us_tok);
    }

    printf("\ntest_glm_expert_bank: %s (%d failure%s)\n",
           failures ? "FAIL" : "OK", failures, failures == 1 ? "" : "s");
    free(pk); free(bk); free(ref); free(got); free(xh); free(wh);
    ds4_gpu_glm_expert_bank_free();
    return failures ? 1 : 0;
}
