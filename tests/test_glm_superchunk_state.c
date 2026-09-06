/*
 * Unit test for the expanded-expert super-chunk exit-status bookkeeping
 * (consultant prefill addendum 38).
 *
 * A super-chunk prefill walks a run of chunks LAYER-major, so at any point in
 * the middle the KDA conv/recurrent state and the DSA indexer tail ring are
 * advanced for the layers already walked and not for the rest.  On a cancel or
 * a failure the driver restores them from bank_state_backup.  If that restore
 * itself fails, the graph state matches no prefix of the prompt: the outcome
 * must NOT be reported as a clean cancellation whose checkpoint may be
 * resumed.  ds4_glm_superchunk_exit_status() turns (raw outcome, restore ok?)
 * into the status the caller acts on, and
 * ds4_glm_superchunk_checkpoint_valid() says whether the live checkpoint may
 * survive it.  Both are pure integer bookkeeping and are exercised here.
 *
 * WHAT THIS DOES NOT COVER: that the GPU-side restore actually reproduces the
 * pre-super-chunk state.  Only a model run establishes that; see the cancel
 * and checkpoint-restore cases of the E2 lifecycle suite.
 *
 * Build/run: make superchunk-state-test
 */

#include <stdio.h>

enum {
    SC_STATE_LOST = -3,
    SC_CANCELLED  = -2,
    SC_FAILED     = -1,
    SC_NOT_TAKEN  =  0,
    SC_DONE       =  1
};

int ds4_glm_superchunk_exit_status(int raw, int restore_ok);
int ds4_glm_superchunk_checkpoint_valid(int status, int checkpoint_len);
int ds4_glm_superchunk_is_interrupt(int status);

static int failures;

static void eq(const char *what, int got, int want) {
    if (got != want) {
        printf("  FAIL %s: got %d, want %d\n", what, got, want);
        failures++;
    }
}

int main(void) {
    /* A completed or declined super-chunk never consults the restore. */
    eq("done, restore ok", ds4_glm_superchunk_exit_status(SC_DONE, 1), SC_DONE);
    eq("done, restore failed", ds4_glm_superchunk_exit_status(SC_DONE, 0), SC_DONE);
    eq("not taken", ds4_glm_superchunk_exit_status(SC_NOT_TAKEN, 0), SC_NOT_TAKEN);

    /* A cancel or a failure whose restore succeeded keeps its own meaning. */
    eq("cancel, restored", ds4_glm_superchunk_exit_status(SC_CANCELLED, 1), SC_CANCELLED);
    eq("failure, restored", ds4_glm_superchunk_exit_status(SC_FAILED, 1), SC_FAILED);

    /* ...and one whose restore FAILED becomes STATE_LOST.  This is the case
     * addendum 38 names: without it a cancellation would be reported with a
     * checkpoint the graph state no longer matches. */
    eq("cancel, restore failed", ds4_glm_superchunk_exit_status(SC_CANCELLED, 0), SC_STATE_LOST);
    eq("failure, restore failed", ds4_glm_superchunk_exit_status(SC_FAILED, 0), SC_STATE_LOST);

    /* Checkpoint survival. */
    eq("done keeps checkpoint", ds4_glm_superchunk_checkpoint_valid(SC_DONE, 0), 1);
    eq("cancel keeps a non-empty checkpoint",
       ds4_glm_superchunk_checkpoint_valid(SC_CANCELLED, 4096), 1);
    eq("cancel with an empty checkpoint has nothing to keep",
       ds4_glm_superchunk_checkpoint_valid(SC_CANCELLED, 0), 0);
    eq("failure drops the checkpoint",
       ds4_glm_superchunk_checkpoint_valid(SC_FAILED, 4096), 0);
    eq("STATE_LOST drops the checkpoint even when it is long",
       ds4_glm_superchunk_checkpoint_valid(SC_STATE_LOST, 62174), 0);
    eq("STATE_LOST drops an empty checkpoint too",
       ds4_glm_superchunk_checkpoint_valid(SC_STATE_LOST, 0), 0);

    /* Only a restored cancellation is reported as an interruption; a lost
     * state is an error, so the caller returns a failure and the session is
     * re-primed rather than resumed. */
    eq("cancel is an interrupt", ds4_glm_superchunk_is_interrupt(SC_CANCELLED), 1);
    eq("STATE_LOST is not an interrupt", ds4_glm_superchunk_is_interrupt(SC_STATE_LOST), 0);
    eq("failure is not an interrupt", ds4_glm_superchunk_is_interrupt(SC_FAILED), 0);
    eq("done is not an interrupt", ds4_glm_superchunk_is_interrupt(SC_DONE), 0);

    /* The composition the driver and the session actually perform. */
    for (int raw = SC_CANCELLED; raw <= SC_FAILED; raw++) {
        const int lost = ds4_glm_superchunk_exit_status(raw, 0);
        eq("composed: no restore never leaves a resumable checkpoint",
           ds4_glm_superchunk_checkpoint_valid(lost, 62174), 0);
        eq("composed: no restore is never an interrupt",
           ds4_glm_superchunk_is_interrupt(lost), 0);
    }

    printf("glm super-chunk state test: %s (%d failure%s)\n",
           failures ? "FAIL" : "OK", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
