#ifndef DS4_DFLASH_CLOCK_H
#define DS4_DFLASH_CLOCK_H
#include "ds4_dflash_budget.h"
#include <time.h>

/* Subtract integer monotonic ticks before conversion. Fast-math may reassociate
 * (floating_seconds_end - floating_seconds_start) * 1000 and manufacture a
 * negative residual even for an empty span. Invalid clocks still fail closed. */
static inline uint64_t dflash_clock_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0 || ts.tv_sec < 0 ||
        ts.tv_nsec < 0 || ts.tv_nsec >= 1000000000L ||
        (uint64_t)ts.tv_sec > (UINT64_MAX - UINT64_C(999999999)) / UINT64_C(1000000000))
        return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}
static inline double dflash_clock_ms(uint64_t start, uint64_t end) {
    if (!start || !end || end < start || end - start > DS4_DFLASH_BUDGET_LIMIT)
        return -1.0;
    return (double)(end - start) / 1e6;
}
static inline double dflash_clock_elapsed_ms(uint64_t start) {
    return dflash_clock_ms(start, dflash_clock_now_ns());
}
static inline bool dflash_clock_account_ns(uint64_t start, uint64_t end,
        uint64_t quantum_ns, uint64_t *out) {
    if (!start || !end || end < start || !quantum_ns ||
        quantum_ns > DS4_DFLASH_BUDGET_LIMIT || end - start > DS4_DFLASH_BUDGET_LIMIT)
        return false;
    *out = end == start ? quantum_ns : end - start;
    return true;
}
#endif
