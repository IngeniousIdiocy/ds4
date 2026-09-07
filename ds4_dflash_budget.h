#ifndef DS4_DFLASH_BUDGET_H
#define DS4_DFLASH_BUDGET_H
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Integer nanoseconds avoid fast-math NaN comparisons and floating debt drift.
 * A request exceeding this deliberately generous arithmetic limit fails closed. */
#define DS4_DFLASH_BUDGET_LIMIT (UINT64_MAX / 102u)
typedef struct {
    bool active, parked;
    uint64_t actual_ns, serial_ns, serial_min_ns, speculative_tokens;
    uint64_t escrow_ns, pending_serial_ns;
    uint32_t pending_rows, bundle_calls, probes, overruns;
    bool pending_optional;
} ds4_dflash_budget;

/* Preserve representation inspection under Clang -ffast-math: an ordinary
 * memcpy bitcast may be folded back into a finite floating predicate. */
static inline uint64_t dflash_budget_f64_bits(double value) {
    volatile uint64_t bits;
    memcpy((void *)&bits, &value, sizeof(bits));
    return bits;
}

static inline bool dflash_budget_ns(double seconds, bool round_up, uint64_t *out) {
    const uint64_t bits = dflash_budget_f64_bits(seconds);
    if ((bits >> 63) || (bits & UINT64_C(0x7ff0000000000000)) ==
        UINT64_C(0x7ff0000000000000) || seconds <= 0.0) return false;
    if (seconds >= (double)DS4_DFLASH_BUDGET_LIMIT / 1e9) return false;
    const double ns = seconds * 1e9;
    if (ns >= (double)DS4_DFLASH_BUDGET_LIMIT) return false;
    uint64_t n = (uint64_t)ns;
    if (round_up && (double)n < ns) n++;
    if (!n || n > DS4_DFLASH_BUDGET_LIMIT) return false;
    *out = n;
    return true;
}
/* Bookkeeping can complete inside one clock tick. Charge that reported
 * quantum for a valid zero interval; evaluator samples and reserves still use
 * the strictly-positive conversion above. Invalid clock metadata fails closed. */
static inline bool dflash_budget_account_ns(double seconds, uint64_t quantum_ns,
                                            uint64_t *out) {
    if (!quantum_ns || quantum_ns > DS4_DFLASH_BUDGET_LIMIT) return false;
    const uint64_t bits = dflash_budget_f64_bits(seconds);
    if ((bits & UINT64_C(0x7fffffffffffffff)) == 0) {
        *out = quantum_ns;
        return true;
    }
    return dflash_budget_ns(seconds, true, out);
}
static inline void dflash_budget_begin(ds4_dflash_budget *b) {
    memset(b, 0, sizeof(*b));
    b->active = true;
}
static inline bool dflash_budget_add(uint64_t *total, uint64_t n) {
    if (n > DS4_DFLASH_BUDGET_LIMIT || *total > DS4_DFLASH_BUDGET_LIMIT - n)
        return false;
    *total += n;
    return true;
}
static inline void dflash_budget_charge(ds4_dflash_budget *b, uint64_t ns) {
    if (!ns || !dflash_budget_add(&b->actual_ns, ns)) {
        b->actual_ns = DS4_DFLASH_BUDGET_LIMIT;
        b->parked = true;
        b->escrow_ns = 0;
    }
}
/* Host accounting between the two bundled calls consumes the same escrow. */
static inline void dflash_budget_account(ds4_dflash_budget *b, uint64_t ns) {
    const uint64_t held = b->escrow_ns;
    dflash_budget_charge(b, ns);
    if (held) {
        if (ns >= held || b->parked) {
            b->parked = true;
            b->overruns++;
            b->escrow_ns = 0;
        } else b->escrow_ns = held - ns;
    }
}
static inline bool dflash_budget_reference(const ds4_dflash_budget *b, uint64_t *ref) {
    if (b->serial_min_ns && b->speculative_tokens >
        DS4_DFLASH_BUDGET_LIMIT / b->serial_min_ns) return false;
    *ref = b->serial_ns;
    return dflash_budget_add(ref, b->speculative_tokens * b->serial_min_ns);
}
static inline uint64_t dflash_budget_credit(const ds4_dflash_budget *b) {
    uint64_t ref;
    if (!dflash_budget_reference(b, &ref)) return 0;
    const uint64_t allowed = ref + ref / 50u;
    return allowed > b->actual_ns ? allowed - b->actual_ns : 0;
}
/* A pending returned block is never implicitly acknowledged. */
static inline void dflash_budget_before_call(ds4_dflash_budget *b) {
    if (b->pending_rows) {
        b->parked = true;
        b->pending_rows = 0;
        b->escrow_ns = 0;
    }
}
static inline bool dflash_budget_reserve(ds4_dflash_budget *b, uint64_t estimate_ns) {
    if (!b->active || b->parked || b->pending_rows || !estimate_ns ||
        estimate_ns > DS4_DFLASH_BUDGET_LIMIT) return false;
    if (b->escrow_ns) return b->bundle_calls == 1u;
    if (dflash_budget_credit(b) < estimate_ns) return false;
    b->escrow_ns = estimate_ns;
    b->bundle_calls = 0;
    b->probes++;
    return true;
}
/* Charge every completed attempt, including failures. Keep escrow only for a
 * refresh followed by its first proposal; it cannot fund repeated refreshes. */
static inline void dflash_budget_finish(ds4_dflash_budget *b, uint64_t actual_ns,
        uint64_t serial_ns, uint32_t rows, bool optional, bool need_proposal) {
    if (!b->active) return;
    dflash_budget_charge(b, actual_ns);
    if (optional) {
        b->bundle_calls++;
        if (!b->escrow_ns || actual_ns > b->escrow_ns) {
            b->parked = true;
            b->overruns++;
            b->escrow_ns = 0;
        } else {
            b->escrow_ns -= actual_ns;
            if (!need_proposal) b->escrow_ns = 0;
            else if (b->bundle_calls != 1u || !b->escrow_ns) b->parked = true;
        }
    }
    if (!rows || (!optional && (!serial_ns || serial_ns > actual_ns))) b->parked = true;
    if (b->parked) b->escrow_ns = 0;
    b->pending_rows = rows;
    b->pending_serial_ns = optional ? 0 : serial_ns;
    b->pending_optional = optional;
}
static inline void dflash_budget_ack(ds4_dflash_budget *b, uint32_t emitted, bool done) {
    if (!b->active) return;
    if (emitted > b->pending_rows) b->parked = true;
    else if (emitted) {
        if (b->pending_optional) {
            if (!b->serial_min_ns || !dflash_budget_add(&b->speculative_tokens, emitted))
                b->parked = true;
        } else if (emitted != 1u || b->pending_rows != 1u ||
                   !dflash_budget_add(&b->serial_ns, b->pending_serial_ns)) {
            b->parked = true;
        } else if (!b->serial_min_ns || b->pending_serial_ns < b->serial_min_ns) {
            /* Reprice earlier speculative credit if a cheaper plain serial
             * sample arrives. Cold capture timing never enters this reference. */
            b->serial_min_ns = b->pending_serial_ns;
        }
    }
    uint64_t ref;
    if (!dflash_budget_reference(b, &ref) || b->actual_ns > ref + ref / 50u) {
        b->parked = true;
    }
    b->pending_rows = 0;
    b->pending_serial_ns = 0;
    if (b->parked || done) b->escrow_ns = 0;
    if (done) b->active = false;
}
#endif
