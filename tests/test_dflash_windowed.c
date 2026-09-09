#include <assert.h>
#include <stdio.h>
#include "../ds4_dflash_adaptive.h"

static void feedback(ds4_dflash_adaptive *a, double ms, unsigned verified,
                     unsigned accepted, unsigned consumed) {
    a->pending_chosen = 7;
    a->pending_verified = verified;
    a->pending_accepted = accepted;
    a->pending_rows = accepted + 1;
    a->pending_ms = ms;
    dflash_adaptive_window_feedback(a, consumed);
}

int main(void) {
    unsetenv("DS4_DFLASH_WINDOWED");
    ds4_dflash_adaptive_config c = dflash_adaptive_profile_config(false);
    assert(c.windowed && !c.loss_meter && !c.savings_retry && !c.early_recovery);
    assert(c.n_min == 4 && c.n_start == 7 && c.n_max == 7 && c.p_min == .75f &&
           c.min_serial_tokens == 16 && c.retry);
    assert(dflash_adaptive_config_valid(&c));
    assert(!dflash_adaptive_profile_config(true).windowed);
    ds4_dflash_adaptive_config invalid = c;
    invalid.min_serial_tokens = 0;
    assert(!dflash_adaptive_config_valid(&invalid));
    invalid = c; invalid.retry = false;
    assert(!dflash_adaptive_config_valid(&invalid));
    ds4_dflash_adaptive a;
    dflash_adaptive_begin(&a, c);
    a.serial_ms = 27;
    a.actual_ns = 1000000000; /* No accumulated savings are required. */
    feedback(&a, 250, 7, 7, 8);
    assert(!a.skip_remaining && a.window_calls == 1);
    feedback(&a, 130, 7, 5, 6);
    assert(!a.skip_remaining);
    feedback(&a, 130, 7, 5, 6);
    assert(a.engaged && !a.skip_remaining && !a.window_calls);
    for (unsigned episode=0; episode<5; ++episode) {
        for (unsigned i=0; i<3; ++i) feedback(&a, 40, 0, 0, 1);
        unsigned expected = episode < 4 ? 16u << episode : 128u;
        assert(a.skip_remaining == expected && !a.engaged);
        a.skip_remaining = 0; /* Simulate consumed serial cooldown. */
    }
    a.skip_remaining = 128;
    dflash_adaptive_reasoning(&a, true);
    const uint64_t debt = a.actual_ns;
    dflash_adaptive_reasoning(&a, false);
    assert(!a.skip_remaining && !a.bad_run && !a.window_calls && a.actual_ns == debt);
    feedback(&a, 140, 7, 7, 3); /* Partial consumer ACK is not a full attempt. */
    assert(!a.window_calls);
    feedback(&a, 400, 7, 0, 1);
    feedback(&a, 400, 7, 0, 1);
    feedback(&a, 400, 7, 0, 1);
    assert(a.skip_remaining == 16 && !a.engaged);
    bool calibration = false;
    a.target_ms[7] = 1000;
    assert(dflash_adaptive_economic(&a, 7, &calibration));
    setenv("DS4_DFLASH_WINDOWED", "0", 1);
    const ds4_dflash_adaptive_config ledger = dflash_adaptive_profile_config(false);
    assert(!ledger.windowed && ledger.loss_meter && ledger.savings_retry &&
           dflash_adaptive_config_valid(&ledger));
    setenv("DS4_DFLASH_WINDOWED", "1", 1);
    assert(dflash_adaptive_profile_config(false).windowed);
    unsetenv("DS4_DFLASH_WINDOWED");
    puts("DFlash windowed feedback: all checks passed");
}
