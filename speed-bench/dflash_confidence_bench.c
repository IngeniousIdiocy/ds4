/* Isolate the CPU work used to normalize a seven-position DFlash block. */

#include "ds4_dflash_confidence.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { ROWS = 7, VOCAB = 154880, ITERS = 40 };
static volatile double sink;

static double seconds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int scan_row(const float *row, float *best, float *second) {
    float row_best = row[0];
    float row_second = -3.402823466e38f;
    if (!dflash_confidence_f32_finite(row_best)) return 0;
    for (size_t i = 1; i < VOCAB; i++) {
        const float value = row[i];
        if (!dflash_confidence_f32_finite(value)) return 0;
        if (value > row_best) {
            row_second = row_best;
            row_best = value;
        } else if (value > row_second) {
            row_second = value;
        }
    }
    *best = row_best;
    *second = row_second;
    return 1;
}

static double scalar_partition(const float *row, float best) {
    double sum = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        sum += (double)expf(row[i] - best);
    }
    return sum;
}

static double scratch_partition(const float *row, float best, float *scratch) {
#if DS4_DFLASH_CONFIDENCE_ACCELERATE
    const float shift = -best;
    int count = VOCAB;
    vDSP_vsadd(row, 1, &shift, scratch, 1, VOCAB);
    vvexpf(scratch, scratch, &count);
    double sum = 0.0;
    for (size_t i = 0; i < VOCAB; i++) sum += (double)scratch[i];
    return sum;
#else
    (void)scratch;
    return scalar_partition(row, best);
#endif
}

int main(void) {
    float *rows = malloc((size_t)ROWS * VOCAB * sizeof(*rows));
    float *scratch = malloc((size_t)VOCAB * sizeof(*scratch));
    if (!rows || !scratch) {
        free(scratch);
        free(rows);
        return 2;
    }

    uint32_t state = UINT32_C(123456789);
    for (size_t i = 0; i < (size_t)ROWS * VOCAB; i++) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        rows[i] = ((float)((state >> 8) & UINT32_C(0xffff)) /
                   65535.0f - 0.5f) * 24.0f;
    }
    float best[ROWS], second[ROWS];
    double max_partition_relative_error = 0.0;
    double max_probability_absolute_error = 0.0;
    for (int row = 0; row < ROWS; row++) {
        const float *values = rows + (size_t)row * VOCAB;
        if (!scan_row(values, &best[row], &second[row])) return 3;
        const double scalar = scalar_partition(values, best[row]);
        const double accelerated = scratch_partition(
            values, best[row], scratch);
        const double partition_error = fabs(scalar - accelerated) / scalar;
        const double probability_error =
            fabs(1.0 / scalar - 1.0 / accelerated);
        if (partition_error > max_partition_relative_error) {
            max_partition_relative_error = partition_error;
        }
        if (probability_error > max_probability_absolute_error) {
            max_probability_absolute_error = probability_error;
        }
        sink += scalar + accelerated;
    }

    const double start_scan = seconds_now();
    for (int iteration = 0; iteration < ITERS; iteration++) {
        for (int row = 0; row < ROWS; row++) {
            float row_best, row_second;
            scan_row(rows + (size_t)row * VOCAB, &row_best, &row_second);
            sink += row_best + row_second;
        }
    }
    const double end_scan = seconds_now();
    for (int iteration = 0; iteration < ITERS; iteration++) {
        for (int row = 0; row < ROWS; row++) {
            sink += scalar_partition(
                rows + (size_t)row * VOCAB, best[row]);
        }
    }
    const double end_scalar = seconds_now();
    for (int iteration = 0; iteration < ITERS; iteration++) {
        for (int row = 0; row < ROWS; row++) {
            sink += scratch_partition(
                rows + (size_t)row * VOCAB, best[row], scratch);
        }
    }
    const double end_scratch = seconds_now();

    const double scan_ms = (end_scan - start_scan) * 1000.0 / ITERS;
    const double scalar_ms = (end_scalar - end_scan) * 1000.0 / ITERS;
    const double scratch_ms = (end_scratch - end_scalar) * 1000.0 / ITERS;
    printf("rows=%d vocab=%d scan=%.3fms scalar_partition=%.3fms "
           "scratch_partition=%.3fms speedup=%.2fx\n",
           ROWS, VOCAB, scan_ms, scalar_ms, scratch_ms,
           scalar_ms / scratch_ms);
    printf("partition_relative_error=%.9g probability_absolute_error=%.9g "
           "accelerate=%d sink=%.3g\n",
           max_partition_relative_error, max_probability_absolute_error,
           DS4_DFLASH_CONFIDENCE_ACCELERATE, sink);
    free(scratch);
    free(rows);
    return 0;
}
