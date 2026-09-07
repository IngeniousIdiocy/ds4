#include "../ds4_dflash_clock.h"
#include "../ds4_dflash_adaptive.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    /* Large uptime, same tick and a one-nanosecond difference: subtraction
     * must occur before conversion even with production -ffast-math. */
    volatile uint64_t ticks[] = {UINT64_C(1234567890123456), UINT64_C(1234567890123456),
                                UINT64_C(1234567890123457)};
    const double empty = dflash_clock_ms(ticks[0], ticks[1]);
    assert(dflash_budget_f64_bits(empty) == 0);
    assert(dflash_clock_ms(ticks[0], ticks[2]) > 0.0);
    assert(dflash_clock_ms(ticks[2], ticks[0]) < 0.0);
    assert(dflash_clock_ms(0, ticks[0]) < 0.0);
    uint64_t ns = 0;
    assert(dflash_clock_account_ns(ticks[0], ticks[1], 1, &ns) && ns == 1);
    assert(!dflash_clock_account_ns(ticks[2], ticks[0], 1, &ns));
    assert(!dflash_clock_account_ns(ticks[0], ticks[1], 0, &ns));
    ds4_dflash_adaptive_config c = {1, 7, 3, .75f, false, true, 512, .01f, .03f, true, true, 0, true};
    ds4_dflash_adaptive a;
    dflash_adaptive_begin(&a, c);
    ds4_dflash_adaptive_trace trace = {.operation="refresh", .draft_ms=empty,
        .serial_ms=dflash_clock_ms(ticks[0], ticks[0] + UINT64_C(25000000))};
    dflash_adaptive_finish(&a, UINT64_C(25000001), 1, &trace);
    dflash_adaptive_ack(&a, 1, false);
    assert(!a.invalid && a.calls == 1 && a.drafted == 0 && a.consumed == 1);
    assert(dflash_adaptive_limit(&a) == 7);
    char printed[64];
    snprintf(printed, sizeof(printed), "draft_ms=%.6f", trace.draft_ms);
    assert(!strcmp(printed, "draft_ms=0.000000"));
    trace.draft_ms = dflash_clock_ms(ticks[2], ticks[0]);
    dflash_adaptive_finish(&a, 1, 1, &trace);
    assert(a.invalid); /* guard still rejects invalid timer metadata */
    puts("dflash clock tests passed");
}
