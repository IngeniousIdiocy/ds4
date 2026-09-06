#define _POSIX_C_SOURCE 200809L
#include "../ds4_dflash_budget.h"
#include <assert.h>
#include <stdio.h>
#include <time.h>
#define MS UINT64_C(1000000)
static const uint64_t U = UINT64_C(500000000);
static void serial(ds4_dflash_budget *b, uint64_t ms) {
    dflash_budget_before_call(b);
    dflash_budget_finish(b, ms * MS + 10000u, ms * MS, 1, false, false);
    dflash_budget_ack(b, 1, false);
}
static void funded(ds4_dflash_budget *b) {
    dflash_budget_begin(b);
    for (unsigned i = 0; i < 1021; i++) serial(b, 25);
    assert(!b->parked && dflash_budget_credit(b) >= U);
}
int main(void) {
    ds4_dflash_budget b = {0}, other = {0};
    assert(!dflash_budget_reserve(&b, U)); /* untracked caller */
    dflash_budget_begin(&b);
    for (unsigned i = 0; i < 24; i++) {
        serial(&b, 25);
        assert(!dflash_budget_reserve(&b, U));
    }
    assert(b.probes == 0); /* a requested future length cannot mint credit */

    funded(&b);
    assert(dflash_budget_reserve(&b, U));
    dflash_budget_finish(&b, 44 * MS, 0, 1, true, true);
    dflash_budget_ack(&b, 1, false);
    assert(dflash_budget_credit(&b) < U && b.escrow_ns == 456 * MS);
    dflash_budget_account(&b, 10000u);
    assert(b.escrow_ns == 456 * MS - 10000u);
    assert(dflash_budget_credit(&b) >= b.escrow_ns);
    assert(dflash_budget_reserve(&b, U)); /* refresh escrow prevents starvation */
    dflash_budget_finish(&b, 150 * MS, 0, 8, true, false);
    dflash_budget_ack(&b, 3, true); /* EOS/stop after three consumed rows */
    assert(b.speculative_tokens == 4 && b.escrow_ns == 0 && !b.active);
    uint64_t ref = 0;
    assert(dflash_budget_reference(&b, &ref));
    assert(ref == (1021u + 4u) * 25u * MS); /* never credit the five unused rows */

    funded(&b);
    uint64_t before = b.actual_ns;
    assert(dflash_budget_reserve(&b, U));
    dflash_budget_finish(&b, 44 * MS, 0, 1, true, true);
    dflash_budget_ack(&b, 0, true); /* cancelled before caller consumes anything */
    assert(b.actual_ns == before + 44 * MS && b.speculative_tokens == 0);
    assert(!b.active && !b.escrow_ns);

    funded(&b);
    before = b.actual_ns;
    assert(dflash_budget_reserve(&b, U));
    dflash_budget_finish(&b, U + 1, 0, 8, true, false);
    dflash_budget_ack(&b, 8, false);
    assert(b.parked && b.overruns == 1 && b.actual_ns == before + U + 1);
    assert(!dflash_budget_reserve(&b, U)); /* savings never excuse an overrun */

    funded(&b);
    before = b.actual_ns;
    assert(dflash_budget_reserve(&b, U));
    dflash_budget_finish(&b, 90 * MS, 0, 0, true, false); /* failed verify */
    assert(b.parked && b.actual_ns == before + 90 * MS && b.pending_rows == 0);

    funded(&b);
    assert(dflash_budget_reserve(&b, U));
    dflash_budget_finish(&b, 44 * MS, 0, 1, true, true);
    dflash_budget_ack(&b, 1, false);
    assert(dflash_budget_reserve(&b, U));
    dflash_budget_finish(&b, 44 * MS, 0, 1, true, true);
    assert(b.parked && !b.escrow_ns); /* cannot refresh forever */

    funded(&b);
    dflash_budget_finish(&b, 25 * MS, 25 * MS, 1, false, false);
    uint64_t observed = b.serial_ns;
    dflash_budget_before_call(&b); /* forgotten ACK: discard, never infer */
    assert(b.parked && b.serial_ns == observed && !b.pending_rows);
    dflash_budget_begin(&other);
    serial(&other, 20);
    assert(!other.parked && other.serial_ns == 20 * MS && other.probes == 0);
    dflash_budget_begin(&b); /* new request on reused session */
    assert(!b.actual_ns && !b.serial_ns && !b.serial_min_ns && !b.parked && !b.probes);

    funded(&b);
    assert(dflash_budget_reserve(&b, U));
    dflash_budget_finish(&b, 100 * MS, 0, 8, true, false);
    dflash_budget_ack(&b, 9, false);
    assert(b.parked && b.speculative_tokens == 0);

    funded(&b);
    assert(dflash_budget_reserve(&b, U));
    dflash_budget_finish(&b, 100 * MS, 0, 8, true, false);
    dflash_budget_ack(&b, 8, false);
    serial(&b, 1); /* cheaper real sample reprices all old speculative credit */
    assert(b.serial_min_ns == MS);
    assert(dflash_budget_reference(&b, &ref));
    assert(ref == b.serial_ns + 8 * MS);

    funded(&b);
    assert(!dflash_budget_reserve(&b, 0));
    assert(!dflash_budget_reserve(&b, UINT64_MAX));
    dflash_budget_finish(&b, UINT64_MAX, 0, 0, true, false);
    assert(b.parked && b.actual_ns == DS4_DFLASH_BUDGET_LIMIT);
    b.serial_ns = DS4_DFLASH_BUDGET_LIMIT;
    b.speculative_tokens = UINT64_MAX;
    assert(!dflash_budget_reference(&b, &ref));
    assert(!dflash_budget_credit(&b));

    uint64_t ns = 0;
    assert(dflash_budget_ns(0.5, true, &ns) && ns == U);
    assert(!dflash_budget_ns(0.0, true, &ns));
    assert(!dflash_budget_ns(-0.5, true, &ns));
    assert(!dflash_budget_ns(1e100, true, &ns));
    const uint64_t bad[] = {UINT64_C(0x7ff8000000000001), UINT64_C(0x7ff0000000000000),
                            UINT64_C(0xfff0000000000000), UINT64_C(0x8000000000000000)};
    for (unsigned i = 0; i < sizeof(bad)/sizeof(bad[0]); i++) {
        double x; memcpy(&x, &bad[i], sizeof(x));
        assert(!dflash_budget_ns(x, true, &ns));
    }
    assert(dflash_budget_account_ns(0.0, 1000u, &ns) && ns == 1000u);
    dflash_budget_begin(&b);
    dflash_budget_account(&b, ns);
    assert(!b.parked && b.actual_ns == 1000u);
    assert(!dflash_budget_ns(0.0, false, &ns)); /* serial samples remain invalid */
    assert(!dflash_budget_reserve(&b, 0));
    assert(!dflash_budget_account_ns(0.0, 0, &ns));
    assert(!dflash_budget_account_ns(0.0, UINT64_MAX, &ns));
    assert(!dflash_budget_account_ns(-1.0, 1000u, &ns));
    for (unsigned i = 0; i < 3; i++) {
        double x; memcpy(&x, &bad[i], sizeof(x));
        assert(!dflash_budget_account_ns(x, 1000u, &ns));
    }

    /* Real host-clock regression: use the production ACK and accounting
     * conversion, including equal endpoints. Do not demand any fixed speed. */
    struct timespec resolution;
    assert(clock_getres(CLOCK_MONOTONIC, &resolution) == 0);
    const uint64_t quantum = (uint64_t)resolution.tv_sec * 1000000000u +
        (uint64_t)resolution.tv_nsec;
    unsigned zero_intervals = 0;
    for (unsigned i = 0; i < 20000; i++) {
        dflash_budget_begin(&b);
        b.pending_rows = 1; b.pending_serial_ns = 25 * MS; b.actual_ns = 25 * MS;
        struct timespec start, end;
        assert(clock_gettime(CLOCK_MONOTONIC, &start) == 0);
        dflash_budget_ack(&b, 1, false);
        assert(clock_gettime(CLOCK_MONOTONIC, &end) == 0);
        const double seconds = ((double)end.tv_sec + (double)end.tv_nsec / 1e9) -
            ((double)start.tv_sec + (double)start.tv_nsec / 1e9);
        assert(dflash_budget_account_ns(seconds, quantum, &ns));
        if (seconds == 0) { zero_intervals++; assert(ns == quantum); }
        dflash_budget_account(&b, ns);
        assert(b.actual_ns == 25 * MS + ns && b.actual_ns != DS4_DFLASH_BUDGET_LIMIT);
    }
    printf("dflash accounting clock: quantum=%llu ns, zero intervals=%u/20000\n",
        (unsigned long long)quantum, zero_intervals);
    puts("dflash request credit/escrow/EOS/cancel/overflow: OK");
    return 0;
}
