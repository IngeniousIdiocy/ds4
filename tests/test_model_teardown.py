#!/usr/bin/env python3
"""CPU-only teardown regression; --source-only skips the tiny C compilation.

Runs the production engine_close/model_close bodies against lifetime-checking
mocks and real page mappings/file descriptors. Metal ordering is source-checked;
this does not execute or emulate GPU commands or prove driver behavior.
"""
import argparse
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def function(source, name):
    match = re.search(r"^[\w *]+\b" + name + r"\([^;]*?\)\s*\{", source, re.M)
    assert match, name
    start = source.index("{", match.start())
    depth = 1
    end = start + 1
    # These selected functions have no braces inside string/comment literals.
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def ordered(source, *parts):
    positions = [source.index(part) for part in parts]
    assert positions == sorted(positions), parts


def source_checks(c, metal):
    close = function(c, "ds4_engine_close")
    ordered(close, "ds4_threads_shutdown();", "ds4_gpu_prepare_cleanup();",
            "ds4_gpu_tensor_free", "dflash_pool_free();",
            "metal_graph_free_prefill_workspace", "ds4_gpu_cleanup();",
            "model_close(&e->dflash_model)", "model_close(&e->mtp_model)",
            "model_close(&e->vision_model)", "model_close(&e->model)")
    assert "e->tp.active || e->tp.slab" in close  # failed TP bind still owns storage
    prepare = function(metal, "ds4_gpu_prepare_cleanup")
    ordered(prepare, "if (!g_initialized) return;",
            "ds4_gpu_queue_keepalive_stop_thread();", "ds4_gpu_tp_shutdown();",
            "ds4_gpu_drain_for_cleanup();")
    assert "ds4_gpu_init(" not in prepare  # early engine failure never initializes Metal
    keepalive = function(metal, "ds4_gpu_queue_keepalive_thread")
    ordered(keepalive, "last_cb = cb;", "[cb commit];",
            "[last_cb waitUntilCompleted];")
    drain = function(metal, "ds4_gpu_drain_for_cleanup")
    ordered(drain, "ds4_gpu_close_batch_encoder();", "[g_batch_cb commit];",
            "[g_batch_cb waitUntilCompleted];", "ds4_gpu_wait_pending_command_buffers")
    shutdown = function(metal, "ds4_gpu_tp_shutdown")
    ordered(shutdown, "pthread_join(g_tp_thread", "ds4_gpu_drain_for_cleanup();",
            "g_tp_poll_buffer = nil;", "g_tp_slab_buffer = nil;")
    cleanup = function(metal, "ds4_gpu_cleanup")
    ordered(cleanup, "ds4_gpu_prepare_cleanup();", "ds4_gpu_glm_expert_bank_free();",
            "ds4_gpu_parallel_ffn_reset_state", "ds4_gpu_stream_expert_pread_pool_shutdown();",
            "ds4_gpu_model_residency_clear();", "ds4_gpu_model_views_clear();",
            "[g_model_buffer_cache removeAllObjects];")
    loader = c[c.index('char dflash_reject[192]'):c.index('/* Also apply explicit optional Q8')]
    ordered(loader, "bool dflash_map_attempted = false;", "dflash_map_attempted = true,",
            "ds4_gpu_set_model_map_range(e->dflash_model.map", "dflash_pool_init(",
            "if (!dflash_map_attempted) model_close(&e->dflash_model);")
    print("PASS teardown source ordering, rejection ownership, no-init drain")


MOCKS = r'''
#define _DARWIN_C_SOURCE 1
#define _DEFAULT_SOURCE 1
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#ifndef __APPLE__
#define __APPLE__ 1
#endif
#define DS4_N_LAYER 2
#define DS4_TP_GATES_PER_LAYER 2
typedef struct { void *kv, *tensors, *map; uint64_t size; int fd; } ds4_model;
typedef struct { bool active; void **out_views, **in_views;
    void **batch_out_views, **batch_in_views; void *zero_vec, *slab; } mock_tp;
typedef struct { bool dflash_ready, shared_prefill_workspace_ready;
    mock_tp tp; int weights, vocab, shared_prefill_workspace, simulated_memory;
    ds4_model model, mtp_model, vision_model, dflash_model;
    void *directional_steering_dirs, *directional_steering_file; } ds4_engine;
static int drained, cleaned, joined, tensor_frees, unmaps, pool_frees;
static void ds4_threads_shutdown(void) { joined = 1; }
static void ds4_gpu_prepare_cleanup(void) { assert(joined); drained = 1; }
static void ds4_gpu_tensor_free(void *p) {
    assert(drained); if (p) { ++tensor_frees; free(p); }
}
static void ds4_gpu_cleanup(void) { assert(drained); cleaned = 1; }
static void dflash_pool_free(void) { assert(drained); ++pool_frees; }
static void glm_dflash_seed_free(void) { assert(drained); }
static void metal_graph_free_prefill_workspace(void *p) { (void)p; assert(drained); }
static void weights_free(void *p) { (void)p; assert(joined); }
static void vocab_free(void *p) { (void)p; }
static void dflash_overhead_report(FILE *f) { (void)f; }
static void ds4_expert_profile_close(void) {}
static void ds4_ssd_memory_lock_release(void *p) { (void)p; }
static void ds4_release_instance_lock(void) {}
static int checked_munmap(void *p, size_t n) {
#ifndef DS4_NO_GPU
    assert(cleaned);  /* mapped backing cannot die while mock views exist */
#endif
    assert(joined); ++unmaps; return munmap(p, n);
}
#define munmap checked_munmap
'''

CASES = r'''
static void setup_model(ds4_model *m, int mapped) {
    m->fd = -1;
    if (!mapped) return;
    m->kv = malloc(8); m->tensors = malloc(8);
    m->fd = open("/dev/null", O_RDONLY); assert(m->fd >= 0);
    m->size = (uint64_t)sysconf(_SC_PAGESIZE);
    m->map = mmap(NULL, (size_t)m->size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(m->map != MAP_FAILED);
}
int main(void) {
    ds4_engine_close(NULL);
    /* Empty/early failure; all models; rejected mapped draft without ready;
     * successful draft; partially allocated TP bind (active remains false). */
    for (int test = 0; test < 5; ++test) {
        drained = cleaned = joined = tensor_frees = unmaps = pool_frees = 0;
        ds4_engine *e = calloc(1, sizeof(*e)); assert(e);
        setup_model(&e->model, test > 0);
        setup_model(&e->mtp_model, test == 1);
        setup_model(&e->vision_model, test == 1);
        setup_model(&e->dflash_model, test > 0);
        e->dflash_ready = test == 3;
        int fds[] = {e->model.fd, e->mtp_model.fd, e->vision_model.fd,
                     e->dflash_model.fd};
#ifndef DS4_NO_GPU
        e->shared_prefill_workspace_ready = test > 0;
        if (test == 4) {
            e->tp.slab = malloc(8);
            e->tp.out_views = calloc(4, sizeof(void *));
            e->tp.out_views[0] = malloc(8);
        }
#endif
        ds4_engine_close(e);
        assert(joined);
        assert(unmaps == (test == 0 ? 0 : test == 1 ? 4 : 2));
        for (unsigned i = 0; i < 4; ++i)
            if (fds[i] >= 0) assert(fcntl(fds[i], F_GETFD) == -1);
#ifndef DS4_NO_GPU
        assert(cleaned && pool_frees == 1);
        assert(tensor_frees == (test == 4 ? 2 : 0));
#endif
    }
    puts("PASS production close CPU mocks: empty, all maps, rejected draft, ready draft, partial TP");
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    c = (ROOT / "ds4.c").read_text()
    source_checks(c, (ROOT / "ds4_metal.m").read_text())
    if args.source_only:
        return
    with tempfile.TemporaryDirectory(prefix="ds4-teardown-") as directory:
        source = pathlib.Path(directory) / "test.c"
        source.write_text(MOCKS + function(c, "model_close") + "\n" +
                          function(c, "ds4_engine_close") + CASES)
        for flags in ([], ["-DDS4_NO_GPU"]):
            exe = pathlib.Path(directory) / ("cpu" if flags else "metal-mock")
            subprocess.run(["cc", "-std=c11", "-Werror=implicit-function-declaration",
                            *flags, str(source), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
