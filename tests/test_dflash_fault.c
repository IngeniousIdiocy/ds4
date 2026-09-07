#include "../ds4_dflash_fault.h"
#include "../ds4_dflash_confidence.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    ds4_dflash_fault session = {0}, independent = {0};
    bool injected = false;
    dflash_fault_begin_request(&session);
    assert(session.requests == 1 && dflash_fault_attempt(&session));
    assert(!dflash_fault_inject_once(false, &injected) && !injected);

    /* Real failure helper invalidates partial per-layer draft KV without
     * touching adjacent storage. Target-state preservation is a GPU gate. */
    struct layer { uint32_t sentinel, len, guard; } layers[3] = {
        {11, 17, 31}, {12, 10, 32}, {13, 10, 33}
    };
    assert(dflash_fault_inject_once(true, &injected));
    assert(dflash_cached_failure(&layers[0].len, sizeof(layers[0]), 3) == -1);
    for (unsigned i = 0; i < 3; i++) {
        assert(layers[i].len == 0 && layers[i].sentinel == 11u + i);
        assert(layers[i].guard == 31u + i);
    }
    dflash_fault_latch(&session, true);
    assert(session.disabled && !session.unsafe && session.failures == 1);

    /* A failing session remains usable for serial work, without retrying
     * the proposer on any later token, rewind, or new request. */
    for (unsigned request = 0; request < 4; request++) {
        if (request) dflash_fault_begin_request(&session);
        for (unsigned token = 0; token < 64; token++) {
            assert(!dflash_fault_attempt(&session));
            assert(dflash_fault_skip(&session));
        }
    }
    assert(session.requests == 4 && session.attempts == 1);
    assert(session.failures == 1 && session.skips == 256);

    /* Independent session state remains eligible; the process-wide once
     * hook is already consumed and cannot poison a healthy new session. */
    dflash_fault_begin_request(&independent);
    assert(!independent.disabled && dflash_fault_attempt(&independent));
    assert(!dflash_fault_inject_once(true, &injected));
    assert(!independent.failures && !independent.skips);

    /* An unsafe drain is not silently converted to an ordinary target
     * evaluation. New requests do not weaken that decision either. */
    ds4_dflash_fault unsafe = {0};
    dflash_fault_begin_request(&unsafe);
    assert(dflash_fault_attempt(&unsafe));
    dflash_fault_latch(&unsafe, false);
    assert(unsafe.unsafe && !dflash_fault_skip(&unsafe));
    dflash_fault_begin_request(&unsafe);
    assert(!dflash_fault_attempt(&unsafe) && !dflash_fault_skip(&unsafe));
    assert(unsafe.attempts == 1 && unsafe.failures == 1 && unsafe.skips == 2);
    puts("dflash session fault latch: PASS");
    return 0;
}
