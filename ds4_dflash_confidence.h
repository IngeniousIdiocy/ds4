#ifndef DS4_DFLASH_CONFIDENCE_H
#define DS4_DFLASH_CONFIDENCE_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
#include <Accelerate/Accelerate.h>
#define DS4_DFLASH_CONFIDENCE_ACCELERATE 1
#else
#define DS4_DFLASH_CONFIDENCE_ACCELERATE 0
#endif

/* Keep the validity checks effective under -ffast-math.  Compiler finite
 * predicates may be folded away when that flag asserts finite inputs. */
static inline int dflash_confidence_f32_bits_finite(uint32_t bits) {
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static inline int dflash_confidence_f32_finite(float value) {
    /* Clang may fold an ordinary memcpy-to-integer bitcast back into a
     * floating finite predicate under -ffast-math.  The volatile integer
     * destination keeps this an inspection of the supplied representation. */
    volatile uint32_t bits;
    memcpy((void *)&bits, &value, sizeof(bits));
    return dflash_confidence_f32_bits_finite(bits);
}

static inline int dflash_confidence_f64_bits_finite(uint64_t bits) {
    return (bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

static inline int dflash_confidence_f64_finite(double value) {
    volatile uint64_t bits;
    memcpy((void *)&bits, &value, sizeof(bits));
    return dflash_confidence_f64_bits_finite(bits);
}

static inline int dflash_confidence_valid(float probability) {
    return dflash_confidence_f32_finite(probability) &&
           probability > 0.0f && probability <= 1.0f;
}

/* Summarize one full-vocabulary logit row.  The probability is the actual
 * normalized probability of the row's top token:
 *
 *     exp(max) / sum_i exp(logit_i)
 *
 * The max-shifted form below is stable, and the partition is accumulated in
 * double so a large vocabulary does not lose most low-probability terms.
 * Any non-finite logit makes the row unusable rather than manufacturing a
 * confidence estimate.  Ties select the lowest index, matching argmax.
 */
static inline int dflash_logits_summary_scratch(
        const float *logits,
        size_t n_logits,
        int *top_index,
        float *top1_top2_margin,
        float *top_probability,
        float *partition_scratch) {
    if (top_index) *top_index = -1;
    if (top1_top2_margin) *top1_top2_margin = -1.0f;
    if (top_probability) *top_probability = -1.0f;
    if (!logits || n_logits == 0) return 0;

    float best = logits[0];
    float second = -3.402823466e+38f;
    int best_index = 0;
    if (!dflash_confidence_f32_finite(best)) return 0;
    for (size_t i = 1; i < n_logits; i++) {
        const float value = logits[i];
        if (!dflash_confidence_f32_finite(value)) return 0;
        if (value > best) {
            second = best;
            best = value;
            best_index = (int)i;
        } else if (value > second) {
            second = value;
        }
    }

    if (top_index) *top_index = best_index;
    if (top1_top2_margin) {
        *top1_top2_margin = n_logits > 1 ? best - second : 3.402823466e+38f;
    }
    if (top_probability) {
        double partition = 0.0;
#if DS4_DFLASH_CONFIDENCE_ACCELERATE
        if (partition_scratch && n_logits <= (size_t)INT32_MAX) {
            const float shift = -best;
            int count = (int)n_logits;
            vDSP_vsadd(logits, 1, &shift, partition_scratch, 1,
                       (vDSP_Length)n_logits);
            vvexpf(partition_scratch, partition_scratch, &count);
            /* Preserve the scalar path's ordered double accumulation.  Only
             * the max shift and per-element exponential are vectorized. */
            for (size_t i = 0; i < n_logits; i++) {
                partition += (double)partition_scratch[i];
            }
        } else
#else
        (void)partition_scratch;
#endif
        {
            for (size_t i = 0; i < n_logits; i++) {
                partition += (double)expf(logits[i] - best);
            }
        }
        if (!dflash_confidence_f64_finite(partition) || partition < 1.0) {
            if (top_index) *top_index = -1;
            if (top1_top2_margin) *top1_top2_margin = -1.0f;
            return 0;
        }
        const float probability = (float)(1.0 / partition);
        if (!dflash_confidence_valid(probability)) {
            if (top_index) *top_index = -1;
            if (top1_top2_margin) *top1_top2_margin = -1.0f;
            return 0;
        }
        *top_probability = probability;
    }
    return 1;
}

static inline int dflash_logits_summary(
        const float *logits,
        size_t n_logits,
        int *top_index,
        float *top1_top2_margin,
        float *top_probability) {
    return dflash_logits_summary_scratch(
        logits, n_logits, top_index, top1_top2_margin, top_probability, NULL);
}

/* Confidence-gated proposals are prefixes: the first low or invalid row ends
 * the usable draft.  An invalid threshold also fails closed. */
static inline int dflash_confident_prefix(
        const float *probabilities,
        int count,
        float min_probability) {
    if (!probabilities || count <= 0 ||
        !dflash_confidence_f32_finite(min_probability) ||
        min_probability < 0.0f || min_probability > 1.0f) {
        return 0;
    }
    int prefix = 0;
    while (prefix < count &&
           dflash_confidence_valid(probabilities[prefix]) &&
           probabilities[prefix] >= min_probability) {
        prefix++;
    }
    return prefix;
}

/* Preserve the legacy trailing-margin diagnostic for opted-in callers.
 * Public full-block callers must retain every valid proposal, even when an
 * inherited process setting requested a margin gate. */
static inline int dflash_legacy_margin_prefix(
        const float *margins, int count, float min_margin, int apply_gate) {
    if (apply_gate && min_margin >= 0.0f) {
        while (count > 0 && margins[count - 1] < min_margin) count--;
    }
    return count;
}

/* A cached GPU pass can append different amounts to different layers before
 * a later layer fails.  Zero every logical length so no partial conditioning
 * can be consumed by the next proposal.  The storage remains allocated and
 * can be reused by a later clean pass. */
static inline int dflash_cached_failure(
        uint32_t *first_len,
        size_t layer_stride,
        uint32_t n_layer) {
    if (first_len && layer_stride >= sizeof(*first_len)) {
        const uint32_t zero = 0;
        unsigned char *p = (unsigned char *)first_len;
        for (uint32_t i = 0; i < n_layer; i++) {
            memcpy(p + (size_t)i * layer_stride, &zero, sizeof(zero));
        }
    }
    return -1;
}

#endif
