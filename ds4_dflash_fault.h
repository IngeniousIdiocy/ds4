#ifndef DS4_DFLASH_FAULT_H
#define DS4_DFLASH_FAULT_H

#include <stdbool.h>
#include <stdint.h>

/* This latch belongs to the allocated session, not its current conditioning
 * generation or request ledger. A new request must not silently retry a
 * failing GPU drafter. Only destroying/recreating the session clears it. */
typedef struct {
    bool disabled, unsafe, request_done;
    uint64_t requests, attempts, failures, skips;
} ds4_dflash_fault;

static inline void dflash_fault_begin_request(ds4_dflash_fault *f) {
    f->requests++;
    f->request_done = false;
}

static inline bool dflash_fault_attempt(ds4_dflash_fault *f) {
    if (f->disabled) return false;
    f->attempts++;
    return true;
}

static inline void dflash_fault_latch(ds4_dflash_fault *f, bool drained) {
    f->disabled = true;
    f->unsafe = !drained;
    f->failures++;
}

static inline bool dflash_fault_skip(ds4_dflash_fault *f) {
    f->skips++;
    return f->disabled && !f->unsafe;
}

/* One process-scoped hook occurrence, consumed only after a real drafter
 * layer has encoded its cache-producing work. The production caller drains
 * that work before returning its ordinary negative failure status. */
static inline bool dflash_fault_inject_once(bool selected, bool *used) {
    if (!selected || *used) return false;
    *used = true;
    return true;
}

#endif
