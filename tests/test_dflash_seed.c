/* Exercise the production prompt-seed lifecycle with bounded GPU I/O mocks. */
#include <stdbool.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_N_EMBD 2u
#define DS4_DFLASH2_MAX_TARGET 8u
typedef struct { unsigned char *data; uint64_t bytes; } ds4_gpu_tensor;
typedef struct {
    bool ready;
    uint32_t n_target, target_layers[8], sliding_window;
} ds4_dflash2_weights;
typedef struct {
    int enabled;
    int fuse_sdn, fuse_replace;
    ds4_gpu_tensor *buf;
    ds4_gpu_tensor *fused_buf;
    uint32_t n_rows, source_row, captured_taps, fused_taps, n_taps, taps[8];
} ds4_glm_dflash_capture;
static ds4_glm_dflash_capture g_glm_dflash_cap;
static struct { uint64_t seed_rows, seed_chunks; double seed_readback_ms; }
    g_dflash_overhead;
static bool fail_read;
static ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->data = calloc(1, (size_t)bytes);
    t->bytes = bytes;
    if (!t->data) { free(t); return NULL; }
    return t;
}
static void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (t) { free(t->data); free(t); }
}
static bool ds4_gpu_tensor_read(ds4_gpu_tensor *t, uint64_t offset,
                                void *out, uint64_t bytes) {
    if (fail_read || !t || offset > t->bytes || bytes > t->bytes - offset)
        return false;
    memcpy(out, t->data + offset, (size_t)bytes);
    return true;
}
static uint32_t glm53_prefill_chunk_tokens(void) { return 128u; }
static double now_sec(void) { static double t; return t += 0.001; }
#include "../ds4_dflash_seed.inc"

static void expect(bool ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}
static void mock_capture(uint32_t start, bool all_taps) {
    float *p = (float *)g_glm_dflash_cap.buf->data;
    for (uint32_t r = 0; r < g_glm_dflash_cap.n_rows; r++) {
        for (uint32_t j = 0; j < 4u; j++)
            p[r * 4u + j] = (float)((start + g_glm_dflash_cap.source_row + r) * 4u + j);
    }
    g_glm_dflash_cap.captured_taps = all_taps ? 3u : 1u;
}
static void expect_rows(uint32_t start, uint32_t rows, uint64_t owner) {
    expect(g_glm_dflash_seed.ring_len == rows, "retained row count");
    expect(g_glm_dflash_seed.owner_gen == owner, "conditioning owner");
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t j = 0; j < 4u; j++)
            expect(g_glm_dflash_seed.ring[r * 4u + j] ==
                   (float)((start + r) * 4u + j), "contiguous causal features");
    }
    expect(!g_glm_dflash_cap.enabled && g_glm_dflash_cap.source_row == 0,
           "capture disarmed and offset reset");
}
int main(void) {
    unsetenv("DS4_DFLASH_CTX_CAP");
    ds4_dflash2_weights w = {.ready=true, .n_target=2u,
        .target_layers={4u, 9u}, .sliding_window=65u};
    glm_dflash_seed_configure(&w, true);
    glm_dflash_seed_bind_owner(7u);
    g_glm_dflash_cap.fuse_sdn = 1;
    g_glm_dflash_cap.fuse_replace = 1;
    g_glm_dflash_cap.fused_buf = (ds4_gpu_tensor *)(uintptr_t)1u;
    g_glm_dflash_cap.fused_taps = 3u;
    expect(glm_dflash_seed_chunk_begin(96u), "first prompt chunk arms");
    expect(g_glm_dflash_cap.source_row == 32u &&
           g_glm_dflash_cap.n_rows == 64u, "capture only the window tail");
    expect(!g_glm_dflash_cap.fuse_sdn && !g_glm_dflash_cap.fuse_replace &&
           !g_glm_dflash_cap.fused_buf && !g_glm_dflash_cap.fused_taps,
           "prompt seed clears serial fusion state");
    expect(g_glm_dflash_seed.buf->bytes == 64u * 4u * sizeof(float),
           "GPU allocation bounded by retained window");
    mock_capture(0u, true);
    g_glm_dflash_cap.fuse_sdn = 1;
    g_glm_dflash_cap.fuse_replace = 1;
    g_glm_dflash_cap.fused_buf = g_glm_dflash_cap.buf;
    g_glm_dflash_cap.fused_taps = 3u;
    glm_dflash_seed_chunk_end(96u, 96u, true);
    expect_rows(32u, 64u, 7u);
    expect(!g_glm_dflash_cap.fuse_sdn && !g_glm_dflash_cap.fuse_replace &&
           !g_glm_dflash_cap.fused_buf && !g_glm_dflash_cap.fused_taps,
           "prompt seed disarm clears serial fusion state");

    expect(glm_dflash_seed_chunk_begin(32u), "partial next chunk arms");
    mock_capture(96u, true);
    glm_dflash_seed_chunk_end(32u, 128u, true);
    expect_rows(64u, 64u, 7u);

    expect(glm_dflash_seed_chunk_begin(32u), "gap chunk arms");
    mock_capture(200u, true);
    glm_dflash_seed_chunk_end(32u, 232u, true);
    expect_rows(200u, 32u, 7u);
    glm_dflash_seed_bind_owner(8u);
    expect(glm_dflash_seed_chunk_begin(32u), "new owner arms");
    mock_capture(232u, true);
    glm_dflash_seed_chunk_end(32u, 264u, true);
    expect_rows(232u, 32u, 8u);

    expect(glm_dflash_seed_chunk_begin(96u), "incomplete taps arm");
    mock_capture(264u, false);
    glm_dflash_seed_chunk_end(96u, 360u, true);
    expect_rows(0u, 0u, 0u);
    expect(glm_dflash_seed_chunk_begin(32u), "failed read arms");
    mock_capture(360u, true);
    fail_read = true;
    glm_dflash_seed_chunk_end(32u, 392u, true);
    expect_rows(0u, 0u, 0u);
    fail_read = false;
    expect(glm_dflash_seed_chunk_begin(96u), "cancelled chunk arms");
    mock_capture(392u, true);
    glm_dflash_seed_chunk_end(96u, 488u, false);
    expect_rows(0u, 0u, 0u);
    expect(glm_dflash_seed_chunk_begin(64u), "capture owner test arms");
    mock_capture(488u, true);
    glm_dflash_seed_bind_owner(9u);
    glm_dflash_seed_chunk_end(64u, 552u, true);
    expect_rows(0u, 0u, 0u);
    setenv("DS4_DFLASH_CTX_CAP", "999999999999999999999999", 1);
    expect(glm_dflash_context_capacity(&w) == 64u, "overflow override refused");
    setenv("DS4_DFLASH_CTX_CAP", "-1", 1);
    expect(glm_dflash_context_capacity(&w) == 64u, "negative override refused");
    setenv("DS4_DFLASH_CTX_CAP", "32oops", 1);
    expect(glm_dflash_context_capacity(&w) == 64u, "trailing override refused");
    setenv("DS4_DFLASH_CTX_CAP", "32", 1);
    glm_dflash_seed_configure(&w, true);
    glm_dflash_seed_bind_owner(10u);
    expect(glm_dflash_seed_chunk_begin(64u), "resized window arms");
    expect(g_glm_dflash_seed.buf->bytes == 32u * 4u * sizeof(float),
           "reconfigure allocates actual new capacity");
    mock_capture(0u, true);
    glm_dflash_seed_chunk_end(64u, 64u, true);
    expect_rows(32u, 32u, 10u);
    unsetenv("DS4_DFLASH_CTX_CAP");
    glm_dflash_seed_configure(&w, false);
    expect(!glm_dflash_seed_chunk_begin(64u), "serial mode does not arm");
    glm_dflash_seed_free();
    expect(!g_glm_dflash_seed.buf && !g_glm_dflash_seed.ring &&
           !g_glm_dflash_cap.buf, "teardown clears owners and buffers");
    puts("DFlash prompt seed lifecycle: PASS");
    return 0;
}
