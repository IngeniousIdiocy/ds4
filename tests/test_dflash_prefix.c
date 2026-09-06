/* Standalone CPU test of production prefix indexing, rollback, and timing.
 * No model, GPU, or ds4.c compilation is needed. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../ds4_glm53_prefix.h"
#include "../ds4_dflash_rollback.h"

static void token_step(int tail[4], int pools[8], unsigned pos, int value) {
    if (pos % 4u != 3u) tail[pos % 4u] = value;
    else pools[pos / 4u] = tail[0] + 10 * tail[1] + 100 * tail[2] + 1000 * value;
}

int main(void) {
    for (unsigned pos = 0; pos < 12; pos++) {
        for (unsigned accepted = 1; accepted <= 8; accepted++) {
            int original[4] = {91, 92, 93, 94}, reference[4], restored[4];
            int refpool[8] = {0}, speculative[8] = {0};
            int raw[12], suffix[12];
            memcpy(reference, original, sizeof original);
            for (unsigned i = 0; i < 12; i++) {
                raw[i] = 100 + (int)i;
                suffix[i] = i < accepted ? raw[i] : 900 + (int)i;
            }
            for (unsigned i = 0; i < accepted; i++)
                token_step(reference, refpool, pos + i, raw[i]);
            for (unsigned slot = 0; slot < 4; slot++) {
                const int src = ds4_glm53_tail_prefix_source(pos, accepted, slot);
                assert(src < (int)accepted);
                restored[slot] = src < 0 ? original[slot] : raw[src];
                assert(restored[slot] == reference[slot]);
                assert(restored[slot] == (src < 0 ? original[slot] : suffix[src]));
            }
            /* Planted off-by-one restore: advancing one row too far must
             * expose the first rejected value to the suffix control when
             * that row writes a tail slot. Keeping this first rejected row
             * unchanged would let this exact error evade the comparison. */
            const unsigned wrong_slot = (pos + accepted) % 4u;
            if (wrong_slot < 3u) {
                const int bad_src = ds4_glm53_tail_prefix_source(
                    pos, accepted + 1u, wrong_slot);
                assert(bad_src == (int)accepted);
                assert(raw[bad_src] != suffix[bad_src]);
            }
            /* A verify may have written future compressed pools. Poison
             * those slots, hide them by floor(frontier/4), then complete the
             * next two pools using the restored tail. Each newly visible
             * pool must have been overwritten with the true continuation. */
            for (unsigned p = 0; p < 8; p++)
                speculative[p] = p < (pos + accepted) / 4u ? refpool[p] : -777;
            for (unsigned i = accepted; i < accepted + 4; i++) {
                token_step(reference, refpool, pos + i, raw[i]);
                token_step(restored, speculative, pos + i, raw[i]);
                for (unsigned p = pos / 4u; p < (pos + i + 1u) / 4u; p++)
                    assert(speculative[p] == refpool[p]);
            }
        }
    }
    for (unsigned pos = 62000; pos < 62004; pos++) {
        for (unsigned n = 1; n <= 8; n++) {
            for (unsigned slot = 0; slot < 4; slot++) {
                assert(ds4_glm53_tail_prefix_source(pos, n, slot) ==
                       ds4_glm53_tail_prefix_source(pos % 4u, n, slot));
            }
            ds4_dflash_rollback_plan p = dflash_rollback_plan(8, n, pos, true,
                                                              true, true, false);
            assert(p.kind == (n == 8 ? DS4_DFLASH_ROLLBACK_NONE :
                                      DS4_DFLASH_ROLLBACK_STEPSNAP));
            assert(p.stepsnap_index == n - 1u && p.frontier_pos == pos + n);
            assert(p.replay_tokens == 0);
            p = dflash_rollback_plan(8, n, pos, true, true, true, true);
            assert(p.kind == DS4_DFLASH_ROLLBACK_REPLAY && p.replay_tokens == n);
        }
    }
    assert(dflash_rollback_plan(8, 8, 10, false, true, false, true).kind ==
           DS4_DFLASH_ROLLBACK_REFUSE);
    assert(dflash_rollback_plan(8, 8, UINT32_MAX - 3u, true, true, true, false).kind ==
           DS4_DFLASH_ROLLBACK_REFUSE);
    assert(dflash_timing_ema(0.0f, 25.0f) == 25.0f); /* first refresh */
    assert(fabsf(dflash_timing_ema(25.0f, 35.0f) - 26.0f) < 1e-5f);
    assert(dflash_timing_ema(25.0f, NAN) == 25.0f);
    assert(dflash_timing_ema(25.0f, -1.0f) == 25.0f);
    assert(dflash_timing_ema(0.0f, 80.0f) == 80.0f); /* reset/new depth */
    puts("dflash prefix indexing/rollback/timing: OK");
    return 0;
}
