/*
 * The caller/filter contract that keeps DFlash greedy-only.
 *
 * The speculative cycle's sampled path signals a rejection by masking the
 * rejected token's logit to -inf and handing the modified vector back as the
 * frontier. Every caller then re-samples that vector with
 * ds4_session_sample(), which re-runs the WHOLE top-k/top-p/min-p filter over
 * the changed logits. Re-filtering a modified vector is not the same as
 * drawing the residual of the original filtered distribution: removing the
 * head of the distribution promotes tokens that the original filter had
 * excluded.
 *
 * This test states that as a property -- "the support the caller re-filters
 * to is contained in the target's original filtered support" -- and shows,
 * on the real sampler, that it does NOT hold. That is why
 * ds4_session_eval_speculative() sends positive temperature down the ordinary
 * path instead of into the cycle.
 *
 * WHEN THE RESIDUAL IS FIXED: computing the residual against the original
 * filtered support, and handing the caller a distribution rather than a
 * masked logit vector, will make the containment hold. The assertions below
 * are written so that fixing it flips them; update this test together with
 * the fix and lift the greedy-only guard in the same change.
 *
 * Build/run: make dflash-sampling-test
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

int ds4_test_sample_top_p_min_p(const float *logits, uint32_t n,
                                float temperature, int top_k, float top_p,
                                float min_p, uint64_t *rng);
int ds4_test_sample_build_probabilities(const float *logits, uint32_t n,
                                        float temperature, int top_k,
                                        float top_p, float min_p,
                                        float *probs);

#define N 3
#define MASK (-1e30f)

static int failures;

static void expect(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

/* The worked example from the audit: target probabilities 0.6 / 0.3 / 0.1. */
static const float k_audit[N] = { 0.6f, 0.3f, 0.1f };
/* A second shape, for the filter whose threshold is relative to the maximum. */
static const float k_flat[N] = { 0.5f, 0.3f, 0.2f };

static void base_logits(float *l, const float *p) {
    for (int i = 0; i < N; i++) l[i] = logf(p[i]);
}

static int support_of(const float *logits, int top_k, float top_p,
                      float min_p, int *support) {
    float probs[N];
    int n = 0;
    if (!ds4_test_sample_build_probabilities(logits, N, 1.0f, top_k, top_p,
                                             min_p, probs)) {
        fprintf(stderr, "FAIL: sample_build_probabilities refused\n");
        failures++;
        return 0;
    }
    for (int i = 0; i < N; i++) {
        if (probs[i] > 0.0f) support[n++] = i;
    }
    return n;
}

static int in_support(const int *support, int n, int tok) {
    for (int i = 0; i < n; i++) {
        if (support[i] == tok) return 1;
    }
    return 0;
}

/* One filter configuration: build the target's support, mask its top token
 * the way the cycle's rejection does, and see what the caller's re-filter
 * admits. */
static void one_filter(const char *label, const float *p, int top_k,
                       float top_p, float min_p) {
    float logits[N];
    int before[N], after[N];
    base_logits(logits, p);

    const int n_before = support_of(logits, top_k, top_p, min_p, before);
    expect(n_before >= 2, "the target support has something to reject into");
    expect(in_support(before, n_before, 0), "the drafted token is in support");

    /* the rejection, exactly as ds4_dflash_glm.inc writes it */
    logits[0] = MASK;

    const int n_after = support_of(logits, top_k, top_p, min_p, after);

    int escaped = -1;
    for (int i = 0; i < n_after; i++) {
        if (after[i] != 0 && !in_support(before, n_before, after[i])) {
            escaped = after[i];
            break;
        }
    }

    /* THE PROPERTY: the caller's re-filtered support must not reach outside
     * the target's original filtered support. It does. */
    if (escaped < 0) {
        fprintf(stderr,
                "FAIL %s: expected the re-filter to escape the original "
                "support; if the residual was fixed, update this test and "
                "lift the greedy-only guard\n", label);
        failures++;
        return;
    }
    printf("  %s: token %d is outside the target support but drawable after "
           "rejection\n", label, escaped);

    /* and it really is drawn, not merely present with zero mass */
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    int hits = 0;
    for (int i = 0; i < 20000; i++) {
        if (ds4_test_sample_top_p_min_p(logits, N, 1.0f, top_k, top_p, min_p,
                                        &rng) == escaped) {
            hits++;
        }
    }
    expect(hits > 0, "the escaped token is actually drawn");
    printf("  %s: drawn %d/20000 times\n", label, hits);
}

int main(void) {
    /* top-k 2: support is {0,1}; after masking 0 the new top two are {1,2}. */
    one_filter("top_k=2", k_audit, 2, 1.0f, 0.0f);
    /* top-p 0.9: nucleus is {0,1} (0.6+0.3); after masking 0 the renormalised
     * nucleus reaches {1,2} (0.75+0.25). */
    one_filter("top_p=0.9", k_audit, 0, 0.9f, 0.0f);
    /* min-p is a floor relative to the maximum, so it escapes whenever
     * removing the head lowers that maximum far enough. With 0.5/0.3/0.2 and
     * min_p 0.5 the support is {0,1} (the floor is 0.25); after masking, the
     * floor is recomputed against 0.3 and 0.2 clears it. The audit's
     * 0.6/0.3/0.1 does NOT escape under min-p -- 0.1 stays below the
     * recomputed 0.15 floor -- which is why it gets its own numbers here. */
    one_filter("min_p=0.5", k_flat, 0, 1.0f, 0.5f);

    /* Greedy is exactly right, which is the path DFlash uses: the frontier
     * after a rejection is the argmax of the remaining logits, and that IS
     * the correct greedy continuation. */
    {
        float logits[N];
        base_logits(logits, k_audit);
        expect(ds4_test_sample_top_p_min_p(logits, N, 0.0f, 0, 1.0f, 0.0f,
                                           NULL) == 0,
               "greedy picks the target argmax");
        logits[0] = MASK;
        expect(ds4_test_sample_top_p_min_p(logits, N, 0.0f, 0, 1.0f, 0.0f,
                                           NULL) == 1,
               "greedy after a rejection picks the next-best token");
    }

    if (failures) {
        fprintf(stderr, "dflash sampling test: %d failure(s)\n", failures);
        return 1;
    }
    printf("dflash sampling test: OK (sampled residual is unsafe; DFlash is "
           "greedy-only)\n");
    return 0;
}
