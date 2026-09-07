/* CPU-only tests for DFlash normalized confidence and fail-closed helpers. */

#include "ds4_dflash_confidence.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        failures++; \
    } \
} while (0)

static void check_normalized_probability(void) {
    const float logits[] = { 1.0f, 2.0f, 3.0f, -4.0f };
    int top = -1;
    float margin = -1.0f;
    float probability = -1.0f;
    const double want = exp(3.0) /
        (exp(1.0) + exp(2.0) + exp(3.0) + exp(-4.0));
    CHECK(dflash_logits_summary(logits, 4, &top, &margin, &probability));
    CHECK(top == 2);
    CHECK(fabsf(margin - 1.0f) < 1e-7f);
    CHECK(fabs((double)probability - want) < 1e-7);

    /* A margin-only proxy would report the same value for both rows, while
     * the full partition correctly makes the crowded row less confident. */
    const float sparse[] = { 4.0f, 3.0f, -20.0f, -20.0f };
    const float crowded[] = { 4.0f, 3.0f, 3.0f, 3.0f };
    float p_sparse = -1.0f, p_crowded = -1.0f;
    CHECK(dflash_logits_summary(sparse, 4, NULL, NULL, &p_sparse));
    CHECK(dflash_logits_summary(crowded, 4, NULL, NULL, &p_crowded));
    CHECK(p_sparse > 0.73f && p_sparse < 0.74f);
    CHECK(p_crowded > 0.47f && p_crowded < 0.48f);

    /* Exercise the production vocabulary scale.  A partial/top-k partition
     * would be orders of magnitude too confident for this uniform row. */
    const size_t vocab = 154880;
    float *wide = (float *)calloc(vocab, sizeof(*wide));
    CHECK(wide != NULL);
    if (wide) {
        float p_wide = -1.0f;
        top = -1;
        margin = -1.0f;
        CHECK(dflash_logits_summary(
            wide, vocab, &top, &margin, &p_wide));
        CHECK(top == 0);
        CHECK(margin == 0.0f);
        CHECK(fabs((double)p_wide - 1.0 / (double)vocab) < 1e-10);
        free(wide);
    }
}

static void check_partition_scratch(void) {
    const size_t count = 4096;
    float *logits = (float *)malloc(count * sizeof(*logits));
    float *scratch = (float *)malloc(count * sizeof(*scratch));
    CHECK(logits != NULL);
    CHECK(scratch != NULL);
    if (!logits || !scratch) {
        free(scratch);
        free(logits);
        return;
    }

    uint32_t state = UINT32_C(123456789);
    for (size_t i = 0; i < count; i++) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        logits[i] = ((float)((state >> 8) & UINT32_C(0xffff)) /
                     65535.0f - 0.5f) * 24.0f;
    }
    int scalar_top = -1, scratch_top = -1;
    float scalar_margin = -1.0f, scratch_margin = -1.0f;
    float scalar_probability = -1.0f, scratch_probability = -1.0f;
    CHECK(dflash_logits_summary(
        logits, count, &scalar_top, &scalar_margin, &scalar_probability));
    CHECK(dflash_logits_summary_scratch(
        logits, count, &scratch_top, &scratch_margin,
        &scratch_probability, scratch));
    CHECK(scratch_top == scalar_top);
    CHECK(scratch_margin == scalar_margin);
    CHECK(fabs((double)scratch_probability - scalar_probability) < 1e-7);

    /* Equal finite maxima retain the lowest-index tie choice.  The extreme
     * finite gap also exercises an underflowing max-shifted exponential. */
    const float extreme[] = { FLT_MAX, -FLT_MAX, FLT_MAX, 0.0f };
    float extreme_probability = -1.0f;
    CHECK(dflash_logits_summary_scratch(
        extreme, 4, &scratch_top, &scratch_margin,
        &extreme_probability, scratch));
    CHECK(scratch_top == 0);
    CHECK(scratch_margin == 0.0f);
    CHECK(fabsf(extreme_probability - 0.5f) < 1e-7f);

    uint32_t nan_bits = UINT32_C(0x7fc00001);
    memcpy(&logits[count / 2], &nan_bits, sizeof(nan_bits));
    scratch_top = 123;
    scratch_probability = 0.5f;
    CHECK(!dflash_logits_summary_scratch(
        logits, count, &scratch_top, NULL, &scratch_probability, scratch));
    CHECK(scratch_top == -1);
    CHECK(scratch_probability == -1.0f);

    free(scratch);
    free(logits);
}

static void check_first_prefix_gate(void) {
    const float confidence[] = { 0.91f, 0.80f, 0.74f, 0.99f };
    CHECK(dflash_confident_prefix(confidence, 4, 0.75f) == 2);
    CHECK(dflash_confident_prefix(confidence, 4, 0.80f) == 2);
    CHECK(dflash_confident_prefix(confidence, 4, 0.92f) == 0);
    CHECK(dflash_confident_prefix(confidence, 4, 0.0f) == 4);

    float invalid[] = { 0.9f, 0.8f, 0.7f };
    uint32_t nan_bits = UINT32_C(0x7fc00001);
    memcpy(&invalid[1], &nan_bits, sizeof(nan_bits));
    CHECK(dflash_confident_prefix(invalid, 3, 0.5f) == 1);
    CHECK(dflash_confident_prefix(invalid, 3, -0.1f) == 0);
}

static void check_invalid_logits(void) {
    float logits[] = { 1.0f, 2.0f, 3.0f };
    uint32_t nan_bits = UINT32_C(0x7fc00001);
    memcpy(&logits[1], &nan_bits, sizeof(nan_bits));
    int top = 123;
    float probability = 0.5f;
    CHECK(!dflash_logits_summary(logits, 3, &top, NULL, &probability));
    CHECK(top == -1);
    CHECK(probability == -1.0f);

    const uint32_t inf_bits = UINT32_C(0x7f800000);
    memcpy(&logits[1], &inf_bits, sizeof(inf_bits));
    CHECK(!dflash_logits_summary(logits, 3, NULL, NULL, &probability));
    CHECK(probability == -1.0f);
}

static void check_full_block_margin_bypass(void) {
    const float margins[] = { 9.0f, 1.0f, 8.0f, 7.0f, 6.0f, 0.2f, 0.1f };
    /* The same inherited threshold trims the legacy diagnostic, while full
     * mode verifies all seven proposals (or the actual capacity-limited
     * count). The interior low margin keeps legacy trailing-only semantics. */
    CHECK(dflash_legacy_margin_prefix(margins, 7, 5.0f, 1) == 5);
    CHECK(dflash_legacy_margin_prefix(margins, 7, 10.0f, 1) == 0);
    CHECK(dflash_legacy_margin_prefix(margins, 7, -1.0f, 1) == 7);
    for (int n = 0; n <= 7; n++) {
        CHECK(dflash_legacy_margin_prefix(margins, n, 5.0f, 0) == n);
        CHECK(dflash_legacy_margin_prefix(margins, n, 10.0f, 0) == n);
        /* Disabled gating does not inspect margin storage at all. */
        CHECK(dflash_legacy_margin_prefix(NULL, n, 10.0f, 0) == n);
    }
}

static void check_bitwise_finiteness(void) {
    const uint32_t cases[][2] = {
        { UINT32_C(0x00000000), 1 }, /* +0 */
        { UINT32_C(0x80000000), 1 }, /* -0 */
        { UINT32_C(0x7f7fffff), 1 }, /* FLT_MAX */
        { UINT32_C(0xff7fffff), 1 }, /* -FLT_MAX */
        { UINT32_C(0x7f800000), 0 }, /* +Inf */
        { UINT32_C(0xff800000), 0 }, /* -Inf */
        { UINT32_C(0x7fc00001), 0 }, /* quiet NaN */
        { UINT32_C(0xffc00001), 0 }, /* negative quiet NaN */
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        float value;
        memcpy(&value, &cases[i][0], sizeof(value));
        CHECK(dflash_confidence_f32_bits_finite(cases[i][0]) ==
              (int)cases[i][1]);
        CHECK(dflash_confidence_f32_finite(value) == (int)cases[i][1]);
        /* Repeat the same word to cover the comparator's equal-word case:
         * equality must not make a NaN or infinity count as finite. */
        float same;
        memcpy(&same, &cases[i][0], sizeof(same));
        CHECK(dflash_confidence_f32_finite(value) ==
              dflash_confidence_f32_finite(same));
    }

    const uint64_t cases64[][2] = {
        { UINT64_C(0x0000000000000000), 1 },
        { UINT64_C(0x7fefffffffffffff), 1 },
        { UINT64_C(0x7ff0000000000000), 0 },
        { UINT64_C(0xfff0000000000000), 0 },
        { UINT64_C(0x7ff8000000000001), 0 },
    };
    for (size_t i = 0; i < sizeof(cases64) / sizeof(cases64[0]); i++) {
        double value;
        memcpy(&value, &cases64[i][0], sizeof(value));
        CHECK(dflash_confidence_f64_bits_finite(cases64[i][0]) ==
              (int)cases64[i][1]);
        CHECK(dflash_confidence_f64_finite(value) == (int)cases64[i][1]);
    }
}

static void check_cached_failure_cleanup(void) {
    struct fake_layer {
        uint32_t before;
        uint32_t len;
        uint32_t after;
    } layer[] = {
        { 11, 7, 12 }, { 21, 5, 22 }, { 31, 2, 32 },
    };
    CHECK(dflash_cached_failure(&layer[0].len, sizeof(layer[0]), 3) == -1);
    for (int i = 0; i < 3; i++) {
        CHECK(layer[i].len == 0);
        CHECK(layer[i].before == (uint32_t)(i * 10 + 11));
        CHECK(layer[i].after == (uint32_t)(i * 10 + 12));
    }
}

int main(void) {
    check_normalized_probability();
    check_partition_scratch();
    check_first_prefix_gate();
    check_invalid_logits();
    check_full_block_margin_bypass();
    check_bitwise_finiteness();
    check_cached_failure_cleanup();
    if (failures) {
        fprintf(stderr, "dflash confidence test: %d failure(s)\n", failures);
        return 1;
    }
    puts("dflash confidence test: OK");
    return 0;
}
