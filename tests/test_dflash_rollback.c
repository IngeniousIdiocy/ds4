/*
 * Unit test for DFlash speculative-rollback bookkeeping.
 *
 * A GLM-5.3 verify forward advances the KDA conv/recurrent state and, on a
 * graph with full DSA layers, the indexer tail K+gate ring.  The per-step
 * snapshot covers both; complete captures restore the accepted prefix, while
 * incomplete captures use the backup and replay (ds4_dflash_glm.inc). Which
 * two happens, how many tokens the replay re-runs, which snapshot row the
 * fast path restores, and where the frontier lands are pure arithmetic in
 * ds4_dflash_rollback.h -- this test drives those functions directly.
 *
 * WHAT THIS DOES NOT COVER: the GPU-side equality of the restored state.
 * That the restored tail ring and KDA buffers actually match a serial
 * continuation can only be established by a model run on the Metal path; see
 * DFLASH-PUBLIC-RECOVERY.md for the forced-rejection check that does it.
 *
 * Build/run: make dflash-rollback-test
 */

#include <stdio.h>
#include <stdint.h>

/* mirrors ds4_dflash_rollback_kind */
enum {
    ROLLBACK_NONE = 0,
    ROLLBACK_STEPSNAP = 1,
    ROLLBACK_REPLAY = 2,
    ROLLBACK_REFUSE = 3,
};

int ds4_test_dflash_state_is_wide(uint64_t spec_bytes, uint64_t kda_bytes);
int ds4_test_dflash_rollback_needs_save(int wide_state, int allow_wide);
int ds4_test_dflash_rollback_can_run(int wide_state, int allow_wide,
                                     int have_backup, int have_stepsnap);
int ds4_test_dflash_session_rewound(uint32_t pos, uint32_t last_pos);
void ds4_test_dflash_rollback_plan(uint32_t L, uint32_t n_committed,
                                   uint32_t pos, int wide_state,
                                   int allow_wide, int have_backup,
                                   int force_replay, int *kind,
                                   uint32_t *stepsnap_index,
                                   uint32_t *replay_tokens,
                                   uint32_t *frontier_pos);

static int failures;

static void expect(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static const char *kind_name(int k) {
    switch (k) {
    case ROLLBACK_NONE: return "none";
    case ROLLBACK_STEPSNAP: return "stepsnap";
    case ROLLBACK_REPLAY: return "replay";
    default: return "refuse";
    }
}

/* Drive one plan and assert every field. */
static void check_plan(const char *label, uint32_t L, uint32_t n_committed,
                       uint32_t pos, int wide, int allow, int backup,
                       int force, int want_kind, uint32_t want_snap,
                       uint32_t want_replay, uint32_t want_frontier) {
    int kind = -1;
    uint32_t snap = 0xffffffffu, replay = 0xffffffffu, frontier = 0xffffffffu;
    ds4_test_dflash_rollback_plan(L, n_committed, pos, wide, allow, backup,
                                  force, &kind, &snap, &replay, &frontier);
    if (kind != want_kind) {
        fprintf(stderr, "FAIL %s: kind=%s want %s\n", label, kind_name(kind),
                kind_name(want_kind));
        failures++;
    }
    if (kind == ROLLBACK_STEPSNAP && snap != want_snap) {
        fprintf(stderr, "FAIL %s: stepsnap_index=%u want %u\n", label, snap,
                want_snap);
        failures++;
    }
    if (replay != want_replay) {
        fprintf(stderr, "FAIL %s: replay_tokens=%u want %u\n", label, replay,
                want_replay);
        failures++;
    }
    if (frontier != want_frontier) {
        fprintf(stderr, "FAIL %s: frontier_pos=%u want %u\n", label, frontier,
                want_frontier);
        failures++;
    }
}

int main(void) {
    /* --- the wide-state predicate --------------------------------------- */
    expect(ds4_test_dflash_state_is_wide(0, 0) == 0, "equal sizes are narrow");
    expect(ds4_test_dflash_state_is_wide(4096, 4096) == 0,
           "equal sizes are narrow");
    expect(ds4_test_dflash_state_is_wide(8192, 4096) == 1,
           "a larger speculative state is wide");

    /* A wide graph must save before the verify; a narrow one need not.
     * With the kill switch set the cycle refuses instead of saving. */
    expect(ds4_test_dflash_rollback_needs_save(1, 1) == 1, "wide saves");
    expect(ds4_test_dflash_rollback_needs_save(0, 1) == 0, "narrow skips save");
    expect(ds4_test_dflash_rollback_needs_save(1, 0) == 0,
           "kill switch skips save");

    /* Admission: wide rolls back by restore + replay, so it needs the kill
     * switch clear and a backup buffer, for fallback; narrow needs a snapshot and nothing else. */
    expect(ds4_test_dflash_rollback_can_run(1, 1, 1, 1) == 1, "wide runs");
    expect(ds4_test_dflash_rollback_can_run(1, 1, 0, 1) == 0,
           "wide without a backup refuses");
    expect(ds4_test_dflash_rollback_can_run(1, 0, 1, 1) == 0,
           "kill switch refuses");
    expect(ds4_test_dflash_rollback_can_run(1, 1, 1, 0) == 1,
           "wide can fall back to replay without a snapshot");
    expect(ds4_test_dflash_rollback_can_run(0, 1, 0, 1) == 1,
           "narrow runs without a backup");
    expect(ds4_test_dflash_rollback_can_run(0, 1, 0, 0) == 0,
           "narrow without a stepsnap refuses");

    /* --- full accept: nothing to undo, on either graph ------------------- */
    check_plan("narrow full accept", 8, 8, 1000, 0, 1, 0, 0,
               ROLLBACK_NONE, 0, 0, 1008);
    check_plan("wide full accept", 8, 8, 1000, 1, 1, 1, 0,
               ROLLBACK_NONE, 0, 0, 1008);
    check_plan("wide full accept L=1", 1, 1, 7, 1, 1, 1, 0,
               ROLLBACK_NONE, 0, 0, 8);

    /* --- forced rejection after 0, 1 and several accepted drafts --------- */
    /* n_committed = accepted drafts + 1 (the first token always commits). */
    check_plan("narrow reject at 0", 8, 1, 1000, 0, 1, 0, 0,
               ROLLBACK_STEPSNAP, 0, 0, 1001);
    check_plan("narrow reject at 1", 8, 2, 1000, 0, 1, 0, 0,
               ROLLBACK_STEPSNAP, 1, 0, 1002);
    check_plan("narrow reject at 5", 8, 6, 1000, 0, 1, 0, 0,
               ROLLBACK_STEPSNAP, 5, 0, 1006);
    check_plan("wide reject at 0", 8, 1, 1000, 1, 1, 1, 0,
               ROLLBACK_STEPSNAP, 0, 0, 1001);
    check_plan("wide reject at 1", 8, 2, 1000, 1, 1, 1, 0,
               ROLLBACK_STEPSNAP, 1, 0, 1002);
    check_plan("wide reject at 5", 8, 6, 1000, 1, 1, 1, 0,
               ROLLBACK_STEPSNAP, 5, 0, 1006);
    /* Rejection one row before the end restores row n_committed-1. */
    check_plan("wide reject at 6 of 8", 8, 7, 1000, 1, 1, 1, 0,
               ROLLBACK_STEPSNAP, 6, 0, 1007);

    /* --- blocks that run to the drafter's maximum block length ---------- */
    /* DS4_DFLASH2_MAX_DRAFT is 7, so L tops out at 8 rows: the last usable
     * per-step snapshot row and the largest replay. */
    check_plan("max block, narrow, last snapshot row", 8, 7, 0, 0, 1, 0, 0,
               ROLLBACK_STEPSNAP, 6, 0, 7);
    check_plan("max block, wide, last partial snapshot", 8, 7, 300000, 1, 1, 1, 0,
               ROLLBACK_STEPSNAP, 6, 0, 300007);
    /* Short blocks: L=2 is one draft, the shortest cycle that can reject. */
    check_plan("L=2 reject", 2, 1, 62000, 1, 1, 1, 0,
               ROLLBACK_STEPSNAP, 0, 0, 62001);
    check_plan("L=2 accept", 2, 2, 62000, 1, 1, 1, 0,
               ROLLBACK_NONE, 0, 0, 62002);

    /* --- degraded configurations refuse rather than half-roll-back ------ */
    check_plan("wide, backup allocation failed", 8, 3, 1000, 1, 1, 0, 0,
               ROLLBACK_REFUSE, 2, 3, 1003);
    check_plan("wide, kill switch set", 8, 3, 1000, 1, 0, 1, 0,
               ROLLBACK_REFUSE, 2, 3, 1003);

    /* --- oracle mode: replay even where the snapshot would be sound ----- */
    check_plan("narrow, forced replay", 8, 3, 1000, 0, 1, 1, 1,
               ROLLBACK_REPLAY, 2, 3, 1003);
    check_plan("narrow, forced replay, full accept", 8, 8, 1000, 0, 1, 1, 1,
               ROLLBACK_REPLAY, 7, 8, 1008);

    /* --- malformed inputs are refusals, never a partial undo ------------ */
    check_plan("zero-length block", 0, 0, 1000, 1, 1, 1, 0,
               ROLLBACK_REFUSE, 0, 0, 1000);
    check_plan("nothing committed", 8, 0, 1000, 1, 1, 1, 0,
               ROLLBACK_REFUSE, 0, 0, 1000);
    check_plan("committed beyond the block", 4, 5, 1000, 1, 1, 1, 0,
               ROLLBACK_REFUSE, 0, 0, 1000);

    /* --- request reuse / rewind ----------------------------------------- */
    /* The backup buffer is written and consumed inside one cycle, so nothing
     * survives a request boundary; what does need resetting is the drafter's
     * context and throttle, keyed on the frontier moving backwards. */
    expect(ds4_test_dflash_session_rewound(0, 0) == 0,
           "a fresh session has not rewound");
    expect(ds4_test_dflash_session_rewound(500, 0) == 0,
           "first cycle of a session has not rewound");
    expect(ds4_test_dflash_session_rewound(500, 500) == 0,
           "standing still is not a rewind");
    expect(ds4_test_dflash_session_rewound(501, 500) == 0,
           "advancing is not a rewind");
    expect(ds4_test_dflash_session_rewound(12, 500) == 1,
           "a slot reused for a shorter prompt has rewound");
    expect(ds4_test_dflash_session_rewound(499, 500) == 1,
           "a one-token thinking-trim rewind counts");

    if (failures) {
        fprintf(stderr, "dflash rollback test: %d failure(s)\n", failures);
        return 1;
    }
    printf("dflash rollback test: OK\n");
    return 0;
}
