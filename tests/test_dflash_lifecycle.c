/*
 * Unit tests for DFlash cycle failure exits and request-state lifecycle.
 *
 * A speculative cycle arms process-global instrumentation (the tap capture,
 * the per-step KDA snapshot) and then mutates per-token target state.  Two
 * things have to hold on every exit: the instrumentation is disarmed, and the
 * session's checkpoint stays valid only if the state was actually put back.
 * The second is a predicate the cycle consults, and it is tested here; the
 * first is structural (one disarm helper, called from every post-arm exit) and
 * is observed on the model through DS4_DFLASH_FAIL.
 *
 * Build/run: make dflash-lifecycle-test
 */

#include <stdio.h>
#include <stdint.h>

int ds4_test_dflash_exit_keeps_checkpoint(int mutated, int state_saved,
                                          int restore_ok);
int ds4_test_dflash_fail_point_selected(const char *spec, const char *name);
int ds4_test_dflash_conditioning_stale(uint64_t bound_gen, uint64_t gen,
                                       uint32_t pos, uint32_t last_pos);
int ds4_test_dflash_seed_ring_usable(uint64_t ring_gen, uint64_t session_gen,
                                     uint32_t ring_len, uint32_t ring_end_pos,
                                     uint32_t pos);

static int failures;

static void expect(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static void exits(void) {
    /* Nothing advanced past the frontier: the checkpoint is untouched, so it
     * survives however the cycle bailed out. Both pre-verify exits (state save
     * failed, no sound rollback) are of this kind. */
    expect(ds4_test_dflash_exit_keeps_checkpoint(0, 0, 0) == 1,
           "unmutated exit keeps the checkpoint");
    expect(ds4_test_dflash_exit_keeps_checkpoint(0, 1, 0) == 1,
           "unmutated exit keeps the checkpoint even with a saved state");

    /* The verify ran: per-token state advanced past the committed frontier.
     * The checkpoint is only usable again if that state was restored. */
    expect(ds4_test_dflash_exit_keeps_checkpoint(1, 1, 1) == 1,
           "restored state keeps the checkpoint");
    expect(ds4_test_dflash_exit_keeps_checkpoint(1, 1, 0) == 0,
           "a failed restore invalidates the checkpoint");
    expect(ds4_test_dflash_exit_keeps_checkpoint(1, 0, 0) == 0,
           "no backup means no restore means no checkpoint");
    expect(ds4_test_dflash_exit_keeps_checkpoint(1, 0, 1) == 0,
           "a restore cannot have happened without a backup");
}

static void fail_points(void) {
    /* Whole-name matching: a prefix or a suffix must not select a point. */
    expect(ds4_test_dflash_fail_point_selected("state_save", "state_save") == 1,
           "exact name selects");
    expect(ds4_test_dflash_fail_point_selected("after_arm,state_save",
                                               "state_save") == 1,
           "name in a list selects");
    expect(ds4_test_dflash_fail_point_selected("state_save, after_verify",
                                               "after_verify") == 1,
           "spaces after commas are tolerated");
    expect(ds4_test_dflash_fail_point_selected("state_save_late",
                                               "state_save") == 0,
           "a longer name is not selected by its prefix");
    expect(ds4_test_dflash_fail_point_selected("late_state_save",
                                               "state_save") == 0,
           "a longer name is not selected by its suffix");
    expect(ds4_test_dflash_fail_point_selected("after_arm", "state_save") == 0,
           "a different name is not selected");
    expect(ds4_test_dflash_fail_point_selected("", "state_save") == 0,
           "an empty spec selects nothing");
    expect(ds4_test_dflash_fail_point_selected(0, "state_save") == 0,
           "an unset spec selects nothing");
    expect(ds4_test_dflash_fail_point_selected("state_save", "") == 0,
           "an unnamed point is never selected");
}

/* Request-state lifecycle: what makes the drafter's conditioning stale, and
 * when the process-global prefill seed ring may be ingested. The scenarios
 * are the ones a frontier comparison alone cannot see. */
static void conditioning(void) {
    /* Same generation, frontier advancing: this is one request generating
     * tokens, and the drafter's warm context is exactly what it should reuse. */
    expect(ds4_test_dflash_conditioning_stale(7, 7, 1000, 999) == 0,
           "an advancing frontier in one generation is not stale");
    expect(ds4_test_dflash_conditioning_stale(7, 7, 1000, 1000) == 0,
           "standing still in one generation is not stale");
    expect(ds4_test_dflash_conditioning_stale(7, 7, 500, 0) == 0,
           "the first cycle of a generation is not stale");

    /* A rewind inside one generation: the frontier moved backwards, so the
     * drafter's rows are for a continuation that no longer exists. */
    expect(ds4_test_dflash_conditioning_stale(7, 7, 999, 1000) == 1,
           "a rewind inside a generation is stale");

    /* Two unrelated prompts of EQUAL length on the same server slot. The
     * frontier is identical before and after, so only the generation shows
     * it: this is the case the old rewind-only predicate missed. */
    expect(ds4_test_dflash_conditioning_stale(7, 8, 4096, 4096) == 1,
           "a different request of the same length is stale");

    /* An old session followed by a LONGER restored prefix on the same slot.
     * The frontier moved forwards, so a rewind test says nothing. */
    expect(ds4_test_dflash_conditioning_stale(7, 8, 300000, 62000) == 1,
           "a longer restored prefix is stale");

    /* A restored prefix that happens to land on the same position. */
    expect(ds4_test_dflash_conditioning_stale(7, 8, 62000, 62000) == 1,
           "a restore to the same position is stale");

    /* Both signals at once. */
    expect(ds4_test_dflash_conditioning_stale(7, 8, 10, 300000) == 1,
           "a new generation with a rewind is stale");
}

static void seed_ring(void) {
    /* The ring is shared by every session in the process. */
    expect(ds4_test_dflash_seed_ring_usable(5, 5, 256, 4096, 4096) == 1,
           "own generation, ending at the frontier, is usable");
    expect(ds4_test_dflash_seed_ring_usable(5, 6, 256, 4096, 4096) == 0,
           "another session's ring is not usable at the same position");
    expect(ds4_test_dflash_seed_ring_usable(5, 5, 256, 4095, 4096) == 0,
           "a ring that does not end at the frontier is not usable");
    expect(ds4_test_dflash_seed_ring_usable(5, 5, 0, 4096, 4096) == 0,
           "an empty ring is not usable");
    expect(ds4_test_dflash_seed_ring_usable(0, 5, 256, 4096, 4096) == 0,
           "a cleared ring is unowned and not usable");
    expect(ds4_test_dflash_seed_ring_usable(5, 0, 256, 4096, 4096) == 0,
           "a session with no generation claims nothing");
    /* The dangerous coincidence: a cancelled prefill left rows ending exactly
     * where a different request now starts decoding. */
    expect(ds4_test_dflash_seed_ring_usable(11, 12, 256, 62000, 62000) == 0,
           "a stale ring ending at the new frontier is still refused");
}

int main(void) {
    exits();
    fail_points();
    conditioning();
    seed_ring();
    if (failures) {
        fprintf(stderr, "dflash lifecycle test: %d failure(s)\n", failures);
        return 1;
    }
    printf("dflash lifecycle test: OK\n");
    return 0;
}
