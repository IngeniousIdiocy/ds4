/*
 * Unit test for the DFlash2 Q8_0 embedding row decode.
 *
 * The GLM-5.3 public target GGUF stores token_embd as Q8_0, and DFlash2 reads
 * the anchor row and the mask row out of that tensor on every proposal
 * (ds4_dflash2.inc, dflash2_embed_row).  This test drives the real reader
 * through ds4_cpu_test_hooks.o and checks it two ways:
 *
 *   1. against an independent Q8_0 decode written here (f16 scale x int8
 *      quant), which covers the arithmetic and the block/row addressing; and
 *   2. against ds4.c's canonical embedding reader embed_token_q8_0(), which
 *      covers the claim that the two production readers agree.
 *
 * Coverage: the full signed quant range including -128/-1/0/1/127, positive,
 * negative, zero, subnormal and near-max f16 scales, several row offsets, a
 * row width that is not a multiple of the 32-value block, and the boundary
 * token ids DFlash2 actually uses (0, vocab-1, and the drafter's mask id).
 *
 * Build/run: make dflash2-embed-q8-test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

void ds4_test_dflash2_embed_row_q8_0(float *out, const void *blob,
                                     uint64_t dim, uint64_t rows, int token);
void ds4_test_embed_token_q8_0_ref(float *out, const void *blob,
                                   uint64_t dim, uint64_t rows, int token);
uint64_t ds4_test_embed_row_bytes_q8_0(uint64_t dim);

static int failures;

static void fail(const char *what, uint64_t dim, int token, uint64_t i,
                 float got, float want) {
    fprintf(stderr,
            "FAIL %s: dim=%llu token=%d index=%llu got=%.9g want=%.9g\n",
            what, (unsigned long long)dim, token, (unsigned long long)i,
            (double)got, (double)want);
    failures++;
}

/* Independent IEEE half -> float, deliberately not ds4.c's. */
static float ref_f16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const int exp = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    float out;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* subnormal half: value = mant * 2^-24 */
            float v = (float)mant * 5.9604644775390625e-8f;
            return (h & 0x8000u) ? -v : v;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* Independent Q8_0 row decode: ceil(dim/32) blocks of {f16 scale, 32 int8}. */
static void ref_row(float *out, const uint8_t *blob, uint64_t dim,
                    int token) {
    const uint64_t blocks = (dim + 31) / 32;
    const uint8_t *row = blob + (uint64_t)token * blocks * 34;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t bits;
        memcpy(&bits, row + b * 34, sizeof(bits));
        const float scale = ref_f16(bits);
        for (uint64_t i = 0; i < 32; i++) {
            const uint64_t idx = b * 32 + i;
            if (idx >= dim) break;
            const int8_t q = (int8_t)row[b * 34 + 2 + i];
            out[idx] = scale * (float)q;
        }
    }
}

/* Scales chosen to span the f16 range: unit, negative, small, large, zero,
 * subnormal, and an ordinary fractional value. */
static const uint16_t k_scales[] = {
    0x3c00, /*  1.0     */
    0xbc00, /* -1.0     */
    0x1400, /*  9.77e-4 */
    0x7bff, /*  65504   */
    0x0000, /*  0       */
    0x0001, /*  subnormal min */
    0x3555, /*  0.3333  */
    0xb266, /* -0.0500  */
    0x0400, /*  smallest normal */
    0xfbff, /* -65504   */
};

static void build(uint8_t *blob, uint64_t dim, uint64_t rows) {
    const uint64_t blocks = (dim + 31) / 32;
    uint32_t rng = 0x1234567u;
    for (uint64_t r = 0; r < rows; r++) {
        uint8_t *row = blob + r * blocks * 34;
        for (uint64_t b = 0; b < blocks; b++) {
            const uint16_t s =
                k_scales[(r + b) % (sizeof(k_scales) / sizeof(k_scales[0]))];
            memcpy(row + b * 34, &s, sizeof(s));
            for (uint64_t i = 0; i < 32; i++) {
                int8_t q;
                /* first block of every row pins the signed extremes */
                if (b == 0 && i < 6) {
                    static const int8_t edge[6] = { -128, -127, -1, 0, 1, 127 };
                    q = edge[i];
                } else {
                    rng = rng * 1664525u + 1013904223u;
                    q = (int8_t)((rng >> 17) & 0xffu);
                }
                row[b * 34 + 2 + i] = (uint8_t)q;
            }
        }
    }
}

static void check_token(const uint8_t *blob, uint64_t dim, uint64_t rows,
                        int token) {
    float *got = malloc((size_t)dim * sizeof(float));
    float *ref = malloc((size_t)dim * sizeof(float));
    float *canon = malloc((size_t)dim * sizeof(float));
    if (!got || !ref || !canon) {
        fprintf(stderr, "FAIL: out of memory\n");
        failures++;
        free(got); free(ref); free(canon);
        return;
    }
    memset(got, 0x7f, (size_t)dim * sizeof(float));
    memset(canon, 0x7f, (size_t)dim * sizeof(float));
    ds4_test_dflash2_embed_row_q8_0(got, blob, dim, rows, token);
    ds4_test_embed_token_q8_0_ref(canon, blob, dim, rows, token);
    ref_row(ref, blob, dim, token);
    for (uint64_t i = 0; i < dim; i++) {
        if (got[i] != ref[i]) fail("dflash2 vs independent", dim, token, i,
                                   got[i], ref[i]);
        if (got[i] != canon[i]) fail("dflash2 vs canonical", dim, token, i,
                                     got[i], canon[i]);
    }
    free(got); free(ref); free(canon);
}

static void run_case(uint64_t dim, uint64_t rows, int mask_token) {
    const uint64_t row_bytes = ((dim + 31) / 32) * 34;
    if (ds4_test_embed_row_bytes_q8_0(dim) != row_bytes) {
        fprintf(stderr, "FAIL: row stride for dim=%llu is %llu, expected %llu\n",
                (unsigned long long)dim,
                (unsigned long long)ds4_test_embed_row_bytes_q8_0(dim),
                (unsigned long long)row_bytes);
        failures++;
    }
    uint8_t *blob = malloc((size_t)(rows * row_bytes));
    if (!blob) {
        fprintf(stderr, "FAIL: out of memory\n");
        failures++;
        return;
    }
    build(blob, dim, rows);
    const int tokens[] = {
        0,                      /* first row */
        1,
        (int)(rows / 2),
        mask_token,             /* the drafter's fixed mask row */
        (int)rows - 2,
        (int)rows - 1,          /* last row */
    };
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        check_token(blob, dim, rows, tokens[i]);
    }
    free(blob);
}

int main(void) {
    /* Target-shaped: 4096-wide rows, block aligned. mask_token mirrors the
     * drafter's 154856 of 154880 -- 24 rows below the top of the table. */
    run_case(4096, 1024, 1024 - 24);
    /* Unaligned width: exercises the partial trailing block. */
    run_case(100, 64, 64 - 24);
    /* Single block. */
    run_case(32, 8, 4);

    if (failures) {
        fprintf(stderr, "dflash2 Q8_0 embed test: %d failure(s)\n", failures);
        return 1;
    }
    printf("dflash2 Q8_0 embed test: OK\n");
    return 0;
}
