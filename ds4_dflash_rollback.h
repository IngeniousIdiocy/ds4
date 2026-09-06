#ifndef DS4_DFLASH_ROLLBACK_H
#define DS4_DFLASH_ROLLBACK_H

#include <stdbool.h>
#include <stdint.h>
#include <math.h>

typedef enum {
    DS4_DFLASH_ROLLBACK_NONE = 0,   /* whole drafted block committed */
    DS4_DFLASH_ROLLBACK_STEPSNAP,   /* complete per-prefix state snapshot */
    DS4_DFLASH_ROLLBACK_REPLAY,     /* restore full state, replay the prefix */
    DS4_DFLASH_ROLLBACK_REFUSE,     /* no sound rollback: decode serially */
} ds4_dflash_rollback_kind;

typedef struct {
    ds4_dflash_rollback_kind kind;
    uint32_t stepsnap_index;  /* STEPSNAP: row boundary to restore */
    uint32_t replay_tokens;   /* REPLAY: committed tokens to re-run */
    uint32_t frontier_pos;    /* position the next token will occupy */
} ds4_dflash_rollback_plan;

/* True when speculative state includes producers beyond KDA. */
static inline bool dflash_state_is_wide(uint64_t spec_state_bytes,
                                 uint64_t kda_state_bytes) {
    return spec_state_bytes != kda_state_bytes;
}

/* A wide-state graph must have a full-state backup saved before the verify,
 * or the cycle cannot roll back and must not run at all. */
static inline bool dflash_rollback_needs_save(bool wide_state, bool allow_wide) {
    return wide_state && allow_wide;
}

/* Wide graphs require a backup so failed/incomplete captures can replay.
 * Narrow graphs may use snapshots alone, invalidating on capture failure. */
static inline bool dflash_rollback_can_run(bool wide_state, bool allow_wide,
                                    bool have_backup, bool have_stepsnap) {
    if (wide_state) return allow_wide && have_backup;
    return have_stepsnap;
}

/* n_committed = accepted drafts + 1 (the always-committed first token).
 * L = drafted block length including that first token. */
static inline ds4_dflash_rollback_plan dflash_rollback_plan(
        uint32_t L,
        uint32_t n_committed,
        uint32_t pos,
        bool wide_state,
        bool allow_wide,
        bool have_backup,
        bool force_replay) {
    ds4_dflash_rollback_plan p;
    p.kind = DS4_DFLASH_ROLLBACK_REFUSE;
    p.stepsnap_index = 0;
    p.replay_tokens = 0;
    p.frontier_pos = pos;
    if (L == 0 || n_committed == 0 || n_committed > L ||
        n_committed > UINT32_MAX - pos) return p;
    p.frontier_pos = pos + n_committed;
    p.replay_tokens = n_committed;
    p.stepsnap_index = n_committed - 1u;
    if (wide_state && (!allow_wide || !have_backup)) return p;
    if (force_replay) {
        /* Check this before full acceptance: the oracle replays all cycles. */
        if (have_backup) p.kind = DS4_DFLASH_ROLLBACK_REPLAY;
        return p;
    }
    /* Batched and serial arithmetic may differ; full acceptance needs no
     * rollback because every row belongs to the committed causal prefix. */
    p.kind = n_committed == L ? DS4_DFLASH_ROLLBACK_NONE
                              : DS4_DFLASH_ROLLBACK_STEPSNAP;
    p.replay_tokens = 0;
    return p;
}

/* Use successful local serial samples, including feature refreshes.
 * Invalid timing never poisons the controller. */
static inline float dflash_timing_ema(float previous, float sample) {
    if (!isfinite(sample) || sample <= 0.0f) return previous;
    return previous > 0.0f && isfinite(previous) ?
        0.9f * previous + 0.1f * sample : sample;
}

#endif
