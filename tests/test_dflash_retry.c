#include "../ds4_dflash_adaptive.h"
#include "../ds4_dflash_history.h"
#include <assert.h>
#include <stdio.h>

static bool near(double a, double b) { return a - b < 0.0001 && b - a < 0.0001; }

static void step(ds4_dflash_adaptive *a, double ms, ds4_dflash_adaptive_trace t,
                 unsigned returned, unsigned consumed, bool done) {
    dflash_adaptive_before_call(a);
    t.funded = t.chosen && dflash_adaptive_funded_retry(a, 0.0);
    dflash_adaptive_finish(a, (uint64_t)(ms * 1e6), returned, &t);
    dflash_adaptive_ack(a, consumed, done);
}

int main(void) {
    ds4_dflash_adaptive_config c = {1, 7, 3, .75f, false, true, 512, .01f, .03f, true, false, 0, true, false};
    ds4_dflash_adaptive a;
    ds4_dflash_adaptive_trace serial = {.serial_ms = 25};
    ds4_dflash_adaptive_trace zero = {.chosen = 7, .proposed = 7,
        .serial_ms = 25, .draft_ms = 20};
    ds4_dflash_adaptive_trace skip = {.serial_ms = 25, .skipped = true};
    ds4_dflash_adaptive_trace good = {.chosen = 7, .proposed = 7, .candidate = 7,
        .verified = 7, .accepted = 7, .draft_ms = 20, .verify_ms = 80};
    bool calibration;

    /* Serial history is completed work, not promised future consumption.
     * With ample historical work the retry meter itself controls cadence. */
    dflash_adaptive_begin(&a, c);
    for (unsigned i = 0; i < 100; i++) step(&a, 25, serial, 1, 1, false);
    step(&a, 45, zero, 1, 1, false);
    assert(a.skip_remaining >= 80 && a.skip_remaining <= 81 && !a.probe_only);
    const uint32_t waiting = a.skip_remaining;
    for (unsigned i = 0; i < waiting; i++) {
        assert(dflash_adaptive_skip(&a));
        step(&a, 25, skip, 1, 1, false);
    }
    assert(!dflash_adaptive_skip(&a) && a.reprobe && a.skipped_steps == waiting);
    step(&a, 100, good, 8, 8, false);
    assert(a.engaged && !a.bad_run && !a.skip_remaining);
    step(&a, 45, zero, 1, 1, false);
    assert(!a.skip_remaining); /* one isolated miss in a useful run */
    step(&a, 45, zero, 1, 1, false);
    assert(a.skip_remaining >= 160 && !a.engaged);

    /* Cold negative evidence trips the soft ledger immediately. The entire
     * attempt is retained, including the required serial anchor evaluation. */
    dflash_adaptive_begin(&a, c);
    step(&a, 45, zero, 1, 1, false);
    assert(a.probe_only && a.probe_entries == 1);
    assert(!a.recovery_used && !a.recovery_pending);
    dflash_adaptive_reset_evidence(&a);
    assert(a.probe_only && a.skip_remaining == c.retry_max);
    assert(near(dflash_adaptive_net_ms(&a), 20));
    assert(a.serial_reference_ms == 25 && a.consumed == 1);

    /* EOS/cancel cannot credit the unconsumed speculative suffix. */
    dflash_adaptive_begin(&a, c);
    step(&a, 25, serial, 1, 1, false);
    step(&a, 100, good, 8, 1, true);
    assert(a.consumed == 2 && a.serial_reference_ms == 50);
    assert(near(dflash_adaptive_net_ms(&a), 75) && !a.active);

    /* Seeded speculation gets an immediate attempt, then one actual serial
     * calibration. Accepted but losing short blocks cannot run indefinitely. */
    dflash_adaptive_begin(&a, c);
    ds4_dflash_adaptive_trace short_one = {.chosen = 7, .proposed = 7,
        .candidate = 1, .verified = 1, .accepted = 1, .draft_ms = 25, .verify_ms = 45};
    step(&a, 70, short_one, 2, 2, false);
    assert(a.need_serial && !a.engaged);
    serial.serial_calibration = true;
    step(&a, 25, serial, 1, 1, false);
    assert(!a.need_serial && !a.engaged && !a.probe_only);
    assert(a.recovery_used && a.recovery_pending && !a.skip_remaining);
    assert(dflash_adaptive_limit(&a) == 7);
    const char *floor_source;
    assert(near(dflash_adaptive_loss_floor(&a, &floor_source), 165));
    assert(!strcmp(floor_source, "row-estimate"));
    assert(dflash_adaptive_economic(&a, 1, &calibration));
    serial.serial_calibration = false;

    /* Profitable full-width recovery is possible from probe-only mode. */
    step(&a, 100, good, 8, 8, false);
    assert(!a.probe_only && a.engaged && !a.skip_remaining);
    assert(a.recovery_checks == 1 && !a.recovery_pending);
    assert(near(dflash_adaptive_loss_floor(&a, &floor_source), 80));
    assert(!strcmp(floor_source, "measured"));
    assert(near(dflash_adaptive_net_ms(&a), -80));
    dflash_adaptive_reset_evidence(&a);
    assert(a.recovery_used && a.recovery_checks == 1);
    dflash_adaptive_offer_recovery(&a, 3);
    assert(!a.recovery_pending);

    /* The same opportunity cannot repeat after a failed early check, and
     * the failure gets a capped wait even below the absolute meter floor. */
    dflash_adaptive_begin(&a, c);
    step(&a, 25, serial, 1, 1, false);
    step(&a, 70, short_one, 2, 2, false);
    assert(a.recovery_pending);
    step(&a, 45, zero, 1, 1, false);
    assert(a.recovery_checks == 1 && !a.recovery_pending);
    assert(a.skip_remaining == c.retry_max);
    dflash_adaptive_offer_recovery(&a, 1);
    assert(!a.recovery_pending);
    dflash_adaptive_begin(&a, c);
    assert(!a.recovery_used && !a.recovery_checks);

    /* Measured full-width cost floors the meter; a later accumulating
     * loss eventually crosses it. Terminal partial ACK grants no recovery. */
    step(&a, 25, serial, 1, 1, false);
    ds4_dflash_adaptive_trace bad_full = good;
    bad_full.accepted = 0;
    step(&a, 100, bad_full, 1, 1, false);
    assert(!a.probe_only && near(dflash_adaptive_net_ms(&a), 75));
    step(&a, 45, zero, 1, 1, false);
    assert(a.probe_only && !a.recovery_used);
    dflash_adaptive_begin(&a, c);
    step(&a, 25, serial, 1, 1, false);
    step(&a, 100, good, 8, 1, true);
    assert(!a.recovery_used && !a.recovery_pending);

    /* A frontend may send a separate terminal zero ACK after consuming its
     * last block. It must not repeat a prior recovery or meter decision. */
    dflash_adaptive_begin(&a, c);
    step(&a, 25, serial, 1, 1, false);
    step(&a, 70, short_one, 2, 2, false);
    step(&a, 100, good, 8, 8, false);
    const uint64_t entries = a.probe_entries, checks = a.recovery_checks;
    const uint64_t losses = a.losing_cycles, skipped = a.skipped_steps;
    assert(a.engaged && !a.probe_only && !a.skip_remaining);
    dflash_adaptive_charge(&a, 1000); /* terminal bookkeeping stays charged */
    dflash_adaptive_ack(&a, 0, true);
    assert(!a.active && !a.probe_only && !a.skip_remaining);
    assert(a.probe_entries == entries && a.recovery_checks == checks);
    assert(a.losing_cycles == losses && a.skipped_steps == skipped);
    assert(!a.pending_recovery && !a.pending_calibration && !a.pending_chosen);

    /* Ordinary serial work receives its own measured reference, so a
     * cold sample followed by a warm sample manufactures no jitter debt. */
    dflash_adaptive_begin(&a, c);
    serial.serial_ms = 40;
    step(&a, 40, serial, 1, 1, false);
    serial.serial_ms = 25;
    step(&a, 25, serial, 1, 1, false);
    assert(a.serial_reference_ms == 65 && near(dflash_adaptive_net_ms(&a), 0));
    assert(a.serial_min_ms == 25 && a.serial_ms == 32.5);

    /* Explicit OFF reproduces continuous confidence checking. */
    c.retry = false;
    dflash_adaptive_begin(&a, c);
    for (unsigned i = 0; i < 20; i++) step(&a, 45, zero, 1, 1, false);
    assert(!a.skip_remaining && !a.probe_only && !a.need_serial);

    /* Savings cannot exist before consumption is acknowledged. A good
     * completed block funds later draft checks, not an unconsumed suffix. */
    c.retry = c.savings_retry = true;
    c.early_recovery = false;
    dflash_adaptive_begin(&a, c);
    step(&a, 25, serial, 1, 1, false);
    dflash_adaptive_before_call(&a);
    dflash_adaptive_finish(&a, 100000000, 8, &good);
    assert(dflash_adaptive_savings_ms(&a) == 0);
    dflash_adaptive_ack(&a, 8, false);
    assert(near(dflash_adaptive_savings_ms(&a), 100));
    ds4_dflash_adaptive_trace declined = zero;
    declined.candidate = 1;
    declined.economic = true;
    step(&a, 45, declined, 1, 1, false);
    step(&a, 45, declined, 1, 1, false);
    assert(a.bad_run == 1 && a.nonescalating_declines == 2);
    assert(a.skip_remaining >= 80 && a.skip_remaining <= 81);
    assert(dflash_adaptive_funded_retry(&a, 0));
    assert(!dflash_adaptive_funded_retry(&a, 50)); /* paid setup cannot be borrowed */
    assert(dflash_adaptive_economic(&a, 1, &calibration));
    for (unsigned i = 0; i < 3; i++) {
        assert(!dflash_adaptive_skip(&a));
        step(&a, 45, declined, 1, 1, false);
    }
    assert(a.funded_retries == 3 && a.funded_draft_ms == 60);
    assert(near(dflash_adaptive_savings_ms(&a), 0) && dflash_adaptive_skip(&a));
    assert(a.bad_run == 1); /* confidence declines never doubled the interval */
    ds4_dflash_adaptive_trace rejected = short_one;
    rejected.accepted = 0;
    step(&a, 70, rejected, 1, 1, false);
    assert(a.bad_run == 2 && a.skip_remaining >= 160);

    /* No bank or acceptance history: the proven cold-zero behavior is
     * unchanged. Later serial jitter cannot change already priced verifies. */
    dflash_adaptive_begin(&a, c);
    step(&a, 45, zero, 1, 1, false);
    assert(a.probe_only && a.skip_remaining == c.retry_max);
    assert(!a.funded_retries && !a.nonescalating_declines);
    dflash_adaptive_begin(&a, c);
    serial.serial_ms = 30;
    declined.serial_ms = 30;
    step(&a, 30, serial, 1, 1, false);
    step(&a, 100, good, 8, 8, false);
    step(&a, 50, declined, 1, 1, false);
    step(&a, 50, declined, 1, 1, false);
    assert(dflash_adaptive_funded_retry(&a, 0));
    const double priced_verify = a.reference_verify_ms;
    const double before_jitter = dflash_adaptive_net_ms(&a);
    serial.serial_ms = 10;
    step(&a, 10, serial, 1, 1, false);
    assert(a.reference_verify_ms == priced_verify);
    assert(near(dflash_adaptive_net_ms(&a), before_jitter));
    const double debt = dflash_adaptive_net_ms(&a);
    dflash_adaptive_reset_evidence(&a);
    assert(near(dflash_adaptive_net_ms(&a), debt));
    assert(!dflash_adaptive_funded_retry(&a, 0));

    /* Draft cost is sunk at the verify choice, but fully charged when
     * deciding whether to draft again. A 45ms verify returning two tokens
     * is cheaper than 50ms serial even after a very expensive paid draft. */
    c.savings_retry = false;
    dflash_adaptive_begin(&a, c);
    serial.serial_ms = 25;
    step(&a, 25, serial, 1, 1, false);
    ds4_dflash_adaptive_trace costly_draft = short_one;
    costly_draft.draft_ms = 200;
    step(&a, 245, costly_draft, 2, 2, false);
    assert(dflash_adaptive_economic(&a, 1, &calibration));
    assert(!dflash_adaptive_profitable_history(&a));
    assert(a.losing_cycles == 1 && a.skip_remaining > 0);
    assert(a.draft_ms == 200 && near(dflash_adaptive_net_ms(&a), 195));
    /* Remaining target work can still be too expensive, independent of
     * whether drafting was cheap or already paid. */
    costly_draft.verify_ms = 80;
    step(&a, 280, costly_draft, 2, 2, false);
    assert(!dflash_adaptive_economic(&a, 1, &calibration));

    /* The first seeded width is measured before serial calibration. Keep
     * that direct measurement, but prefer later calibrated evidence when
     * predicting an unseen width instead of perpetuating its cold cost. */
    dflash_adaptive_begin(&a, c);
    ds4_dflash_adaptive_trace cold_five = good;
    cold_five.candidate = cold_five.verified = cold_five.accepted = 5;
    cold_five.verify_ms = 116;
    step(&a, 140, cold_five, 6, 6, false);
    serial.serial_calibration = true;
    step(&a, 25, serial, 1, 1, false);
    serial.serial_calibration = false;
    ds4_dflash_adaptive_trace warm_seven = good;
    warm_seven.verify_ms = 114;
    step(&a, 134, warm_seven, 8, 8, false);
    assert(near(dflash_adaptive_target_cost(&a, 3, &floor_source), 25 + 89.0*3/7));
    assert(!strcmp(floor_source, "row-calibrated"));
    assert(near(dflash_adaptive_target_cost(&a, 5, &floor_source), 116));
    assert(!strcmp(floor_source, "measured"));
    assert(a.draft_ms == 40 && a.actual_ns == 299000000);

    /* A bounded median and exact serial reference cancel serial jitter on
     * both sides, including a rising central estimate. Real extra work stays. */
    dflash_adaptive_begin(&a, c);
    for (unsigned i = 0; i < 30; i++) {
        serial.serial_ms = i & 1u ? 40 : 10;
        step(&a, serial.serial_ms, serial, 1, 1, false);
        assert(near(dflash_adaptive_net_ms(&a), 0));
    }
    assert(a.serial_sample_count == 9 && a.serial_min_ms == 10);
    assert(a.serial_ms == 40 && !a.probe_only);
    step(&a, 42, serial, 1, 1, false);
    assert(near(dflash_adaptive_net_ms(&a), 2)); /* setup/readback remains real */

    /* Explicit bootstrap pricing may correct a cold estimate downward.
     * A later higher median never inflates historical verification credit. */
    dflash_adaptive_begin(&a, c);
    step(&a, 100, good, 8, 8, false);
    assert(a.reference_unpriced_rows == 8 && a.serial_reference_ms == 0);
    serial.serial_ms = 40;
    serial.serial_calibration = true;
    step(&a, 40, serial, 1, 1, false);
    serial.serial_calibration = false;
    assert(a.bootstrap_added_ms == 320 && a.reference_serial_added_ms == 40);
    assert(a.reference_verify_ms == 320 && a.bootstrap_rows == 8);
    serial.serial_ms = 10;
    step(&a, 10, serial, 1, 1, false);
    assert(a.serial_ms == 25 && a.bootstrap_reprice_ms == 120);
    assert(a.reference_verify_ms == 200 && a.bootstrap_repriced_ms == 120);
    const double saved = dflash_adaptive_savings_ms(&a);
    serial.serial_ms = 100;
    step(&a, 100, serial, 1, 1, false);
    assert(a.serial_ms == 40 && a.reference_verify_ms == 200);
    assert(a.bootstrap_reprice_ms == 0 && near(dflash_adaptive_savings_ms(&a), saved));
    const double reference_before = a.serial_reference_ms;
    dflash_adaptive_ack(&a, 0, true);
    assert(a.serial_reference_ms == reference_before);
    assert(!a.reference_serial_added_ms && !a.reference_verify_added_ms);
    assert(!a.bootstrap_added_ms && !a.bootstrap_reprice_ms);

    /* Captures retain chronological rows through wraparound. Lost or
     * discontinuous rows cannot be silently spliced into old draft KV. */
    ds4_dflash_history h = {.cap = 4};
    float ring[8], out[8];
    const float first[] = {1, 11, 2, 12, 3, 13};
    const float next[] = {4, 14, 5, 15, 6, 16};
    assert(dflash_history_append(&h, ring, first, 3, 2, 103));
    assert(dflash_history_append(&h, ring, next, 3, 2, 106));
    dflash_history_read(&h, ring, out, 2);
    assert(h.len == 4 && h.end == 106);
    for (unsigned i = 0; i < 4; i++) assert(out[2*i] == 3+i && out[2*i+1] == 13+i);
    assert(!dflash_adaptive_context_contiguous(106, h.len, h.end, 100));
    assert(dflash_history_append(&h, ring, next, 1, 2, 200));
    assert(h.len == 1 && h.end == 200);
    dflash_history_read(&h, ring, out, 2);
    assert(out[0] == 4 && out[1] == 14);
    puts("dflash measured retry/history/soft ledger: PASS");
    return 0;
}
