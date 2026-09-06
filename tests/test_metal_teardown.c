/* Root-scheduled GPU test: four tiny synthetic file-backed model maps, one
 * flushed batch and one still-open batch, unretained Metal references. No GGUF.
 * Compile/link with the normal ds4_metal.o and ds4_image.o; never runs via make.
 */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }
}

int main(void) {
    enum { MAPS = 4, IN = 256, OUT = 64 };
    const size_t bytes = IN * OUT * sizeof(float);
    void *maps[MAPS] = {0};
    int fds[MAPS];
    ds4_gpu_tensor *views[MAPS] = {0};
    /* Presence semantics; set before the first command-buffer factory call. */
    check(setenv("DS4_METAL_UNRETAINED_COMMAND_BUFFERS", "1", 1) == 0, "set flag");
    ds4_gpu_prepare_cleanup();  /* safe before initialization */
    check(ds4_gpu_init(), "Metal init");
    for (int m = 0; m < MAPS; ++m) {
        char path[] = "/tmp/ds4-teardown-map-XXXXXX";
        fds[m] = mkstemp(path);
        check(fds[m] >= 0, "temp backing");
        check(unlink(path) == 0, "unlink owned temp name");
        check(ftruncate(fds[m], (off_t)bytes) == 0, "backing size");
        maps[m] = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fds[m], 0);
        check(maps[m] != MAP_FAILED, "map backing");
        float *weights = maps[m];
        for (int row = 0; row < OUT; ++row)
            for (int col = 0; col < IN; ++col)
                weights[row * IN + col] = (float)(m * OUT + row + col);
        check(mprotect(maps[m], bytes, PROT_READ) == 0, "read-only backing");
        check(ds4_gpu_set_model_map_range(maps[m], bytes, 0, bytes, bytes), "register map");
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(IN * sizeof(float));
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc(MAPS * OUT * sizeof(float));
    check(x && y, "allocate operands");
    float *xp = ds4_gpu_tensor_contents(x);
    float *yp = ds4_gpu_tensor_contents(y);
    check(xp && yp, "shared operand pointers");
    memset(xp, 0, IN * sizeof(float));
    xp[7] = 1.0f;
    for (int i = 0; i < MAPS * OUT; ++i) yp[i] = -999.0f;
    for (int m = 0; m < MAPS; ++m) {
        views[m] = ds4_gpu_tensor_view(y, m * OUT * sizeof(float), OUT * sizeof(float));
        check(views[m] != NULL, "output view");
    }

    check(ds4_gpu_begin_commands(), "begin first batch");
    for (int m = 0; m < MAPS; ++m) {
        if (m == 2) {
            check(ds4_gpu_flush_commands(), "leave a submitted pending batch");
            check(ds4_gpu_commands_active(), "flush opened the replacement batch");
        }
        check(ds4_gpu_matmul_f32_tensor(views[m], maps[m], bytes, 0,
                                        IN, OUT, x, 1), "encode mapped matvec");
    }
    check(ds4_gpu_commands_active(), "batch must be active before prepare_cleanup");
    ds4_gpu_prepare_cleanup();
    check(!ds4_gpu_commands_active(), "prepare_cleanup closed the batch");
    /* Read the pointer obtained BEFORE dispatch: no read API can accidentally
     * synchronize here and mask a missing teardown drain. */
    for (int m = 0; m < MAPS; ++m)
        for (int row = 0; row < OUT; ++row)
            check(yp[m * OUT + row] == (float)(m * OUT + row + 7),
                  "both pending and open batch results completed");

    for (int m = 0; m < MAPS; ++m) ds4_gpu_tensor_free(views[m]);
    ds4_gpu_tensor_free(y);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();  /* drop residency and all four no-copy model views */
    for (int m = 0; m < MAPS; ++m) {
        check(munmap(maps[m], bytes) == 0, "unmap after cleanup");
        check(close(fds[m]) == 0, "close backing descriptor");
    }
    ds4_gpu_prepare_cleanup();
    ds4_gpu_cleanup();  /* repeated cleanup remains a no-op */
    puts("PASS Metal teardown: 4 maps, pending+open batches, 256 results, unretained references");
    return 0;
}
