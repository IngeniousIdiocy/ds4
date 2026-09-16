/* Bitwise equivalence of kernel_dsv41_rope_quantize against rope + quantize + copy on random rows. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"

static float randn(void) {
    double u = (rand() + 1.0) / (RAND_MAX + 2.0), v = (rand() + 1.0) / (RAND_MAX + 2.0);
    return (float)(sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v));
}

int main(int argc, char **argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 2000;
    srand(argc > 2 ? atoi(argv[2]) : 1);
    if (!ds4_gpu_init()) { fprintf(stderr, "no gpu\n"); return 2; }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(512 * 4), *xa = ds4_gpu_tensor_alloc(512 * 4), *dst = ds4_gpu_tensor_alloc(512 * 4);
    float src[512], a[512], b[512];
    long mism = 0, elems = 0, rows_bad = 0, shown = 0;
    long per_format[4] = {0}, per_parity[2] = {0}, per_rot[2] = {0};
    for (int t = 0; t < trials; t++) {
        const int format = 1 + t % 3;                 /* FP8_E8M0, FP4_E8M0, FP4_E4M3 */
        const uint32_t width = format == 2 ? 128u : 512u;
        const bool compressed = (t / 3) % 2;
        const uint32_t pos = (uint32_t)(rand() % 60000);
        const float scale = (t % 7 == 0) ? 0.02f : (t % 5 == 0) ? 12.0f : 1.0f;
        for (uint32_t i = 0; i < width; i++) src[i] = randn() * scale;
        if (!ds4_gpu_tensor_write(xa, 0, src, width * 4) || !ds4_gpu_tensor_write(x, 0, src, width * 4)) return 3;
        if (!ds4_gpu_dsv41_rope(xa, width, 1, 1, pos, compressed, false) ||
            !ds4_gpu_dsv41_quantize(xa, width, 1, (ds4_v41_activation_format)format)) { fprintf(stderr, "separate path failed\n"); return 4; }
        if (!ds4_gpu_dsv41_rope_quantize(x, dst, 0, width, pos, compressed, (ds4_v41_activation_format)format)) { fprintf(stderr, "fused path failed\n"); return 5; }
        if (!ds4_gpu_synchronize()) return 6;
        if (!ds4_gpu_tensor_read(xa, 0, a, width * 4) || !ds4_gpu_tensor_read(dst, 0, b, width * 4)) return 7;
        bool bad = false;
        for (uint32_t i = 0; i < width; i++) {
            elems++;
            if (memcmp(&a[i], &b[i], 4)) {
                mism++; bad = true; per_format[format]++; per_parity[i & 1]++; per_rot[i >= width - 64u]++;
                if (shown < 8) { shown++; fprintf(stderr, "mismatch t=%d fmt=%d comp=%d pos=%u i=%u sep=%.9g fused=%.9g src=%.9g\n", t, format, compressed, pos, i, a[i], b[i], src[i]); }
            }
        }
        rows_bad += bad;
    }
    printf("trials=%d rows_bad=%ld elems=%ld mismatches=%ld fmt1=%ld fmt2=%ld fmt3=%ld even=%ld odd=%ld rotary=%ld nonrot=%ld\n",
           trials, rows_bad, elems, mism, per_format[1], per_format[2], per_format[3], per_parity[0], per_parity[1], per_rot[1], per_rot[0]);
    return mism ? 1 : 0;
}
