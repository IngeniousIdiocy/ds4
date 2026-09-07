#include "../ds4_dflash_adaptive.h"
#include <assert.h>
#include <stdio.h>

static void cycle(ds4_dflash_adaptive *a, unsigned verified, unsigned accepted,
                  unsigned consumed, bool done) {
    ds4_dflash_adaptive_trace t = {0};
    t.chosen = t.proposed = 7;
    t.verified = verified;
    t.accepted = accepted;
    dflash_adaptive_before_call(a);
    dflash_adaptive_finish(a, UINT64_C(100000000), accepted + 1u, &t);
    dflash_adaptive_ack(a, consumed, done);
}

/* Volatile raw words reach optimized production helpers at runtime; this
 * must not collapse to compile-time tests of literal NAN/INFINITY macros. */
static void check_nonfinite_inputs(void) {
    volatile uint32_t words[] = {UINT32_C(0x7f800000), UINT32_C(0xff800000),
        UINT32_C(0x7fc00001), UINT32_C(0xffc00001), UINT32_C(0x7f800001)};
    for (unsigned i = 0; i < sizeof(words)/sizeof(words[0]); i++) {
        uint32_t word = words[i];
        float value;
        memcpy(&value, &word, sizeof(value));
        assert(!dflash_adaptive_probability(value));
        float confidence[] = {0.99f, value, 0.99f};
        assert(dflash_adaptive_prefix(confidence, 3, 3, 0, 1) == 1);
        assert(dflash_adaptive_prefix(confidence, 1, 1, value, 1) == 0);
        ds4_dflash_adaptive_config base = {1, 7, 3, .75f, false, true, 512, .01f, .03f, true, true, 0, true};
        for (unsigned field = 0; field < 3; field++) {
            ds4_dflash_adaptive_config config = base;
            if (field == 0) config.p_min = value;
            if (field == 1) config.retry_tax = value;
            if (field == 2) config.loss_budget = value;
            ds4_dflash_adaptive a;
            dflash_adaptive_begin(&a, config);
            assert(a.invalid && !dflash_adaptive_limit(&a));
            assert(!dflash_adaptive_funded_retry(&a, 0));
        }
    }
    volatile uint64_t times[] = {UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
        UINT64_C(0x7ff8000000000001), UINT64_C(0xfff8000000000001), UINT64_C(0x7ff0000000000001)};
    for (unsigned i = 0; i < sizeof(times)/sizeof(times[0]); i++) {
        uint64_t word = times[i], ns = 123;
        double value;
        memcpy(&value, &word, sizeof(value));
        assert(!dflash_budget_ns(value, true, &ns) && ns == 123);
        assert(!dflash_budget_account_ns(value, 1, &ns) && ns == 123);
        ds4_dflash_adaptive_config config = {1, 7, 3, .75f, false, true, 512, .01f, .03f, true, true, 0, true};
        ds4_dflash_adaptive direct;
        dflash_adaptive_begin(&direct, config);
        direct.serial_reference_ms = 100;
        direct.skip_remaining = 1;
        direct.draft_estimate_ms = 20;
        assert(dflash_adaptive_funded_retry(&direct, 0));
        assert(!dflash_adaptive_funded_retry(&direct, value));
        dflash_adaptive_serial_sample(&direct, value);
        assert(direct.invalid && !direct.serial_sample_count);
        for (unsigned field = 0; field < 6; field++) {
            ds4_dflash_adaptive a;
            dflash_adaptive_begin(&a, config);
            ds4_dflash_adaptive_trace t = {0};
            double *parts[] = {&t.setup_ms, &t.draft_ms, &t.verify_ms,
                              &t.heads_ms, &t.tail_ms, &t.serial_ms};
            *parts[field] = value;
            dflash_adaptive_finish(&a, 1000, 1, &t);
            dflash_adaptive_ack(&a, 1, true);
            assert(a.invalid && !a.active && a.actual_ns == 1000 && a.consumed == 1);
            assert(!a.serial_sample_count && !a.serial_reference_ms);
            assert(!dflash_adaptive_limit(&a) && !dflash_adaptive_funded_retry(&a, 0));
        }
    }
}

int main(void) {
    check_nonfinite_inputs();
    ds4_dflash_adaptive_config c = {1, 7, 3, 0.75f, true, false, 512, 0.01f, 0.03f, true, false, 0, true};
    ds4_dflash_adaptive a;
    dflash_adaptive_begin(&a, c);
    assert(dflash_adaptive_limit(&a) == 3);
    cycle(&a, 3, 3, 4, false);
    assert(dflash_adaptive_limit(&a) == 4);
    cycle(&a, 4, 1, 2, false);
    assert(dflash_adaptive_limit(&a) == 2);
    cycle(&a, 2, 0, 1, false);
    assert(dflash_adaptive_limit(&a) == 1);
    cycle(&a, 1, 1, 2, false);
    assert(dflash_adaptive_limit(&a) == 2); /* recovery is immediate */
    assert(a.consumed == 9 && a.actual_ns == UINT64_C(400000000));

    /* Only the causal prefix is eligible, even if later confidence recovers. */
    const float p[] = {0.9f, 0.8f, 0.1f, 0.99f, 0.99f};
    assert(dflash_adaptive_prefix(p, 5, 5, 0.75f, 1) == 2);
    assert(dflash_adaptive_prefix(p, 5, 1, 0.75f, 1) == 1);
    assert(dflash_adaptive_prefix(p, 5, 5, 0.75f, 3) == 0);
    float invalid;
    uint32_t nan = UINT32_C(0x7fc00000);
    memcpy(&invalid, &nan, sizeof(invalid));
    assert(dflash_adaptive_prefix(&invalid, 1, 1, 0, 1) == 0);
    assert(!dflash_adaptive_probability(-0.1f));
    assert(!dflash_adaptive_probability(1.1f));
    assert(dflash_adaptive_probability(0) && dflash_adaptive_probability(1));
    assert(dflash_adaptive_context_contiguous(105, 5, 105, 100));
    assert(dflash_adaptive_context_contiguous(105, 5, 105, 0));
    assert(!dflash_adaptive_context_contiguous(105, 5, 104, 100));
    assert(!dflash_adaptive_context_contiguous(105, 5, 105, 99));
    assert(!dflash_adaptive_context_contiguous(3, 5, 3, 0));

    /* EOS/cancel charges the entire attempt and cannot reward unused rows. */
    dflash_adaptive_begin(&a, c);
    cycle(&a, 7, 7, 2, true);
    assert(a.returned == 8 && a.consumed == 2 && a.observations == 0);
    assert(a.actual_ns == UINT64_C(100000000) && !a.active);
    dflash_adaptive_begin(&a, c);
    cycle(&a, 7, 7, 0, true);
    assert(a.consumed == 0 && a.observations == 0 && !a.invalid);

    /* A failed target call cannot disappear from the completed request. */
    ds4_dflash_adaptive_trace t = {0};
    dflash_adaptive_begin(&a, c);
    dflash_adaptive_finish(&a, 1234, 0, &t);
    dflash_adaptive_ack(&a, 0, true);
    assert(a.invalid && a.actual_ns == 1234 && !a.active);

    dflash_adaptive_begin(&a, c);
    dflash_adaptive_finish(&a, 1000, 1, &t);
    dflash_adaptive_before_call(&a);
    assert(a.invalid && !a.consumed && !dflash_adaptive_limit(&a));
    dflash_adaptive_begin(&a, c);
    dflash_adaptive_finish(&a, 1000, 1, &t);
    dflash_adaptive_ack(&a, 2, false);
    assert(a.invalid && !a.consumed);

    /* Conditioning changes discard evidence while retaining request debt. */
    dflash_adaptive_begin(&a, c);
    cycle(&a, 3, 3, 4, false);
    dflash_adaptive_reset_evidence(&a);
    assert(a.actual_ns == UINT64_C(100000000) && a.consumed == 4);
    assert(a.next == 3 && !a.observations);

    c.adaptive = false;
    dflash_adaptive_begin(&a, c);
    cycle(&a, 7, 0, 1, false);
    assert(dflash_adaptive_limit(&a) == 7); /* fixed trained-prefix baseline */
    c.p_min = invalid;
    dflash_adaptive_begin(&a, c);
    assert(a.invalid && dflash_adaptive_limit(&a) == 0);
    c.p_min = 0.75f;
    c.n_start = 8;
    assert(!dflash_adaptive_config_valid(&c));
    dflash_adaptive_charge(&a, UINT64_MAX);
    assert(a.invalid && a.actual_ns == DS4_DFLASH_BUDGET_LIMIT);
    puts("dflash adaptive: PASS");
    return 0;
}
