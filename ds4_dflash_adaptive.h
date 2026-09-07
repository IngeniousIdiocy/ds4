#ifndef DS4_DFLASH_ADAPTIVE_H
#define DS4_DFLASH_ADAPTIVE_H

#include "ds4_dflash_budget.h"

#define DS4_DFLASH_ADAPTIVE_MAX 7u
#define DS4_DFLASH_SERIAL_WINDOW 9u
#define DS4_DFLASH_ENTRY_MAX 64u

/* The drafter retains its trained block geometry. This policy limits only
 * the causal prefix sent to the target verifier. The first proposal is
 * eligible after the configured consumed serial prefix; later retries may spend completed savings. Target-prefix
 * economics remain separate, and all work stays in the request ledger. */
typedef struct {
    uint32_t n_min, n_max, n_start;
    float p_min;
    bool adaptive;
    bool retry;
    uint32_t retry_max;
    float retry_tax;
    float loss_budget;
    bool early_recovery, savings_retry;
    uint32_t min_serial_tokens;
    bool loss_meter;
} ds4_dflash_adaptive_config;

/* Full-block mode reuses the consumed-prefix cost ledger for diagnostics;
 * no confidence, width, retry, entry, savings, or meter decision gates it. */
static inline ds4_dflash_adaptive_config dflash_adaptive_profile_config(bool full_block) {
    const ds4_dflash_adaptive_config full = {1u, 7u, 7u, 0.0f, false, false, 0u,
        0.0f, 0.0f, false, false, 0u, false};
    const ds4_dflash_adaptive_config conservative = {1u, 7u, 3u, .75f, false, true, 512u,
        .01f, .03f, true, true, 0u, true};
    return full_block ? full : conservative;
}

typedef struct {
    ds4_dflash_adaptive_config config;
    bool active, invalid;
    uint32_t next, pending_rows, pending_verified, pending_accepted;
    uint64_t calls, drafted, proposed, verified, accepted, returned, consumed, actual_ns;
    double acceptance, serial_ms, cycle_ms[DS4_DFLASH_ADAPTIVE_MAX + 1u];
    uint32_t observations;
    bool engaged, reprobe, need_serial;
    uint32_t bad_run, skip_remaining, pending_chosen, pending_candidate;
    bool pending_skipped, pending_economic, pending_calibration;
    bool pending_serial, pending_entry;
    uint64_t serial_consumed, entry_serial_tokens;
    double pending_ms, pending_draft_ms;
    double unpriced_ms, unpriced_draft_ms;
    uint32_t unpriced_rows, unpriced_accepted;
    bool recovery_used, recovery_pending, pending_recovery;
    uint64_t recovery_checks;
    double draft_estimate_ms, draft_ema_ms;
    uint64_t funded_retries, nonescalating_declines;
    double funded_draft_ms;
    double target_ms[8], target_units[8], prefix_accept[8];
    uint32_t prefix_observations[8];
    uint64_t zero_prefix, skipped_steps, economic_declines, losing_cycles, calibrations, serial_calibrations;
    bool probe_only;
    uint64_t probe_entries, probe_entry_consumed;
    double serial_reference_ms, reference_step_ms;
    double serial_samples[DS4_DFLASH_SERIAL_WINDOW], serial_min_ms;
    uint32_t serial_sample_count, serial_sample_next;
    double pending_serial_ms, pending_reference_step_ms;
    double reference_serial_ms, reference_verify_ms;
    uint64_t reference_unpriced_rows, bootstrap_rows;
    double bootstrap_step_ms, bootstrap_repriced_ms;
    double reference_serial_added_ms, reference_verify_added_ms;
    double bootstrap_added_ms, bootstrap_reprice_ms, reference_call_step_ms;
    double setup_ms, draft_ms, verify_ms, heads_ms, tail_ms, refresh_ms;
} ds4_dflash_adaptive;

typedef struct {
    const char *operation;
    uint32_t chosen, proposed, verified, accepted, bootstrap_rows;
    uint32_t candidate;
    bool skipped, economic, calibration, serial_calibration, recovery, funded, entry;
    uint64_t serial_consumed_before;
    double funding_bank_ms, funding_cost_ms;
    float confidence_first, confidence_min;
    double setup_ms, draft_ms, verify_ms, heads_ms, tail_ms, serial_ms;
} ds4_dflash_adaptive_trace;

static inline bool dflash_adaptive_probability(float p) {
    volatile uint32_t bits;
    memcpy((void *)&bits, &p, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000) &&
        p >= 0.0f && p <= 1.0f;
}

/* Segment timers may be zero, but cannot supply NaN/Inf/negative credit.
 * Bound them by the same generous range as integer request accounting. */
static inline bool dflash_adaptive_timing(double ms) {
    const uint64_t bits = dflash_budget_f64_bits(ms);
    if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) ||
        ((bits >> 63) && (bits & UINT64_C(0x7fffffffffffffff)))) return false;
    return ms <= (double)DS4_DFLASH_BUDGET_LIMIT / 1e6;
}

static inline bool dflash_adaptive_entry_configure(ds4_dflash_adaptive_config *c,
                                                   const char *setting) {
    if (!setting) return true;
    if (!*setting) return false;
    uint32_t n = 0;
    for (const char *p = setting; *p; p++) {
        if (*p < '0' || *p > '9') return false;
        n = n * 10u + (uint32_t)(*p - '0');
        if (n > DS4_DFLASH_ENTRY_MAX) return false;
    }
    c->min_serial_tokens = n;
    return true;
}

static inline bool dflash_adaptive_config_valid(const ds4_dflash_adaptive_config *c) {
    return c->min_serial_tokens <= DS4_DFLASH_ENTRY_MAX && c->n_min >= 1u && c->n_min <= c->n_start &&
        c->n_start <= c->n_max && c->n_max <= DS4_DFLASH_ADAPTIVE_MAX &&
        dflash_adaptive_probability(c->p_min) && (!c->retry ||
        (c->retry_max >= 1u && c->retry_max <= 4096u &&
         dflash_adaptive_probability(c->retry_tax) &&
         c->retry_tax >= 0.001f && c->retry_tax <= 0.1f &&
         dflash_adaptive_probability(c->loss_budget) &&
         c->loss_budget >= 0.001f && c->loss_budget <= 0.25f));
}

static inline void dflash_adaptive_reset_evidence(ds4_dflash_adaptive *a) {
    a->next = a->config.adaptive ? a->config.n_start : a->config.n_max;
    a->acceptance = 0.0;
    a->observations = 0;
    a->serial_ms = a->reference_step_ms = 0.0;
    a->serial_sample_count = a->serial_sample_next = 0;
    /* Freeze old conditioning's priced reference; never price a lost
     * uncalibrated prefix using measurements from a different context. */
    a->reference_unpriced_rows = a->bootstrap_rows = 0;
    a->bootstrap_step_ms = 0.0;
    a->draft_estimate_ms = a->draft_ema_ms = 0.0;
    a->engaged = a->reprobe = a->need_serial = false;
    a->bad_run = 0;
    /* Conditioning resets discard evidence, never request debt or the spent
     * early opportunity. A probe-only wait must survive those resets too. */
    /* skip_remaining is request policy state and is preserved. */
    memset(a->cycle_ms, 0, sizeof(a->cycle_ms));
    memset(a->target_ms, 0, sizeof(a->target_ms));
    memset(a->target_units, 0, sizeof(a->target_units));
    memset(a->prefix_accept, 0, sizeof(a->prefix_accept));
    memset(a->prefix_observations, 0, sizeof(a->prefix_observations));
}

static inline void dflash_adaptive_begin(ds4_dflash_adaptive *a,
                                        ds4_dflash_adaptive_config config) {
    memset(a, 0, sizeof(*a));
    a->config = config;
    a->active = true;
    a->invalid = !dflash_adaptive_config_valid(&config);
    dflash_adaptive_reset_evidence(a);
}

/* A missing ACK disables optional work; it never assumes the returned
 * suffix was consumed. Diagnostic/accounting faults cannot erase costs. */
static inline void dflash_adaptive_before_call(ds4_dflash_adaptive *a) {
    if (a->pending_rows) {
        a->invalid = true;
        a->pending_rows = 0;
    }
}

static inline void dflash_adaptive_charge(ds4_dflash_adaptive *a, uint64_t ns) {
    if (!ns || !dflash_budget_add(&a->actual_ns, ns)) {
        a->actual_ns = DS4_DFLASH_BUDGET_LIMIT;
        a->invalid = true;
    }
}

static inline bool dflash_adaptive_entry_wait(const ds4_dflash_adaptive *a) {
    return a->active && !a->invalid && !a->drafted &&
        a->serial_consumed < a->config.min_serial_tokens;
}

static inline uint32_t dflash_adaptive_limit(const ds4_dflash_adaptive *a) {
    if (!a->active || a->invalid || dflash_adaptive_entry_wait(a)) return 0;
    if (a->config.retry && (a->probe_only || a->recovery_pending))
        return a->config.n_max;
    uint32_t n = a->config.adaptive ? a->next : a->config.n_max;
    if (a->config.adaptive && a->observations && a->serial_ms > 0.0 &&
        a->cycle_ms[n] > (1.0 + a->acceptance * n) * a->serial_ms)
        n = a->config.n_min;
    return n;
}

static inline uint32_t dflash_adaptive_prefix(const float *confidence,
        uint32_t proposed, uint32_t limit, float p_min, uint32_t n_min) {
    if (!dflash_adaptive_probability(p_min)) return 0;
    uint32_t n = 0;
    while (n < proposed && n < limit &&
           dflash_adaptive_probability(confidence[n]) && confidence[n] >= p_min)
        n++;
    return n >= n_min ? n : 0u;
}

static inline bool dflash_adaptive_context_contiguous(uint32_t frontier,
        uint32_t rows, uint32_t pending_end, uint32_t cache_end) {
    return rows > 0 && rows <= frontier && pending_end == frontier &&
        (!cache_end || cache_end == frontier - rows);
}

static inline double dflash_adaptive_net_ms(const ds4_dflash_adaptive *a) {
    return (double)a->actual_ns / 1e6 - a->serial_reference_ms;
}

static inline double dflash_adaptive_savings_ms(const ds4_dflash_adaptive *a) {
    const double bank = -dflash_adaptive_net_ms(a);
    return bank > 0.0 ? bank : 0.0;
}

/* Savings exist only after consumed-prefix ACK. Current call setup has not
 * yet entered actual_ns, so the caller deducts it when making this choice. */
static inline bool dflash_adaptive_funded_retry(const ds4_dflash_adaptive *a,
                                               double setup_ms) {
    return dflash_adaptive_timing(setup_ms) &&
        a->config.retry && a->config.savings_retry && a->active && !a->invalid &&
        a->skip_remaining && a->draft_estimate_ms > 0.0 &&
        dflash_adaptive_savings_ms(a) - setup_ms >= a->draft_estimate_ms;
}

static inline bool dflash_adaptive_skip(const ds4_dflash_adaptive *a) {
    return a->config.retry && a->active && !a->invalid && a->skip_remaining != 0 &&
        !dflash_adaptive_funded_retry(a, 0.0);
}

/* Use actual target acceptance, never draft softmax as target acceptance.
 * A nearest measured width supplies a simple row-cost estimate when this
 * width has not run. Its one-serial-step intercept avoids pretending short
 * verifier blocks have no fixed cost. Receipts must validate the estimate. */
static inline double dflash_adaptive_target_cost(const ds4_dflash_adaptive *a,
        uint32_t n, const char **source) {
    *source = "unavailable";
    if (!n || n > 7u) return 0.0;
    uint32_t nearest = a->target_ms[n] > 0.0 ? n : 0u;
    bool have_calibrated = false;
    for (uint32_t i = 1; i <= 7; i++)
        if (a->target_ms[i] > 0.0 && a->target_units[i] > 0.0)
            have_calibrated = true;
    if (!nearest) {
        for (uint32_t i = 1; i <= 7; i++) {
            if (!a->target_ms[i] || (have_calibrated && a->target_units[i] <= 0.0)) continue;
            const uint32_t distance = i > n ? i - n : n - i;
            const uint32_t old = nearest > n ? nearest - n : n - nearest;
            if (!nearest || distance < old) nearest = i;
        }
    }
    if (!nearest || (nearest != n && a->serial_ms <= 0.0)) return 0.0;
    *source = nearest == n ? "measured" :
        a->target_units[nearest] > 0.0 ? "row-calibrated" : "row-estimate";
    double raw = a->target_ms[nearest];
    if (a->serial_ms <= 0.0) return raw;
    double units = a->target_units[nearest] > 0.0 ? a->target_units[nearest] :
        raw / a->serial_ms;
    if (nearest != n) units = 1.0 + (units - 1.0) * n / nearest;
    if (units < 1.0) units = 1.0;
    double target = units * a->serial_ms;
    if (nearest != n) {
        raw = a->serial_ms + (raw - a->serial_ms) * n / nearest;
        if (raw < a->serial_ms) raw = a->serial_ms;
    }
    /* A cheaper serial calibration is not evidence that the target verifier
     * itself sped up. Do not discount measured target work on that basis. */
    return target > raw ? target : raw;
}

static inline double dflash_adaptive_loss_floor(const ds4_dflash_adaptive *a,
                                               const char **source) {
    return dflash_adaptive_target_cost(a, a->config.n_max, source);
}

static inline bool dflash_adaptive_economic(const ds4_dflash_adaptive *a,
        uint32_t n, bool *calibration) {
    *calibration = false;
    if (!a->config.retry || !n || n > 7u || a->serial_ms <= 0.0) return true;
    const char *source;
    const double target = dflash_adaptive_target_cost(a, n, &source);
    if (target <= 0.0) { *calibration = true; return true; }
    /* The proposal is already paid. Only remaining target work can be
     * avoided by declining this prefix. Full draft cost still governs the
     * next attempt through ACK profitability, backoff and the request ledger. */
    const double cost = target;
    if (cost >= (n + 1.0) * a->serial_ms) return false;
    if (!a->prefix_observations[n] || a->reprobe) {
        *calibration = true;
        return true;
    }
    return cost < (1.0 + a->prefix_accept[n]) * a->serial_ms;
}

/* A confidence/economic decline did not ask the target to reject anything.
 * Preserve the interval when actual accepted-prefix history still pays at
 * current measured costs; a rejected/losing verified cycle may escalate. */
static inline bool dflash_adaptive_profitable_history(const ds4_dflash_adaptive *a) {
    if (a->serial_ms <= 0.0 || a->draft_estimate_ms <= 0.0) return false;
    for (uint32_t n = 1; n <= 7; n++) {
        if (!a->prefix_observations[n] || a->target_ms[n] <= 0.0) continue;
        const char *source;
        const double cost = a->draft_estimate_ms + dflash_adaptive_target_cost(a, n, &source);
        if (cost < (1.0 + a->prefix_accept[n]) * a->serial_ms) return true;
    }
    return false;
}

static inline void dflash_adaptive_backoff(ds4_dflash_adaptive *a, double optional_ms,
                                           bool escalate) {
    const bool first_bad = a->bad_run == 0;
    if (escalate || first_bad) a->bad_run++;
    if (!escalate) a->nonescalating_declines++;
    /* One isolated miss need not interrupt a demonstrably useful run. */
    if (a->engaged && first_bad) return;
    a->engaged = false;
    if (a->probe_only) {
        a->skip_remaining = a->config.retry_max;
        a->reprobe = false;
        return;
    }
    double steps = a->serial_ms > 0.0 ?
        optional_ms / (a->config.retry_tax * a->serial_ms) : 1.0;
    if (steps < 1.0) steps = 1.0;
    for (uint32_t i = 1; i < a->bad_run && steps < a->config.retry_max; i++)
        steps *= 2.0;
    if (steps > a->config.retry_max) steps = a->config.retry_max;
    a->skip_remaining = (uint32_t)steps;
    if ((double)a->skip_remaining < steps) a->skip_remaining++;
    a->reprobe = false;
}

/* One early opportunity per request, backed by committed target agreement.
 * Losing startup calibration and meter entry share the same allowance. */
static inline void dflash_adaptive_offer_recovery(ds4_dflash_adaptive *a,
                                                 uint32_t accepted) {
    if (!a->config.early_recovery || !accepted || a->recovery_used) return;
    a->recovery_used = a->recovery_pending = true;
    a->skip_remaining = 0;
    a->reprobe = true;
}

/* A bounded central reference resists isolated serial jitter. The minimum
 * remains diagnostic; neither estimate retroactively prices old work up. */
static inline void dflash_adaptive_serial_sample(ds4_dflash_adaptive *a, double ms) {
    if (!dflash_adaptive_timing(ms) || ms <= 0.0) { a->invalid = true; return; }
    if (a->serial_min_ms <= 0.0 || ms < a->serial_min_ms) a->serial_min_ms = ms;
    a->serial_samples[a->serial_sample_next] = ms;
    a->serial_sample_next = (a->serial_sample_next + 1u) % DS4_DFLASH_SERIAL_WINDOW;
    if (a->serial_sample_count < DS4_DFLASH_SERIAL_WINDOW) a->serial_sample_count++;
    double ordered[DS4_DFLASH_SERIAL_WINDOW];
    for (uint32_t i = 0; i < a->serial_sample_count; i++) {
        uint32_t j = i;
        while (j && ordered[j - 1u] > a->serial_samples[i]) {
            ordered[j] = ordered[j - 1u];
            j--;
        }
        ordered[j] = a->serial_samples[i];
    }
    const uint32_t middle = a->serial_sample_count / 2u;
    a->serial_ms = a->serial_sample_count & 1u ? ordered[middle] :
        0.5 * (ordered[middle - 1u] + ordered[middle]);
    a->reference_step_ms = a->serial_ms;
}

static inline void dflash_adaptive_finish(ds4_dflash_adaptive *a,
        uint64_t actual_ns, uint32_t rows, const ds4_dflash_adaptive_trace *t) {
    dflash_adaptive_charge(a, actual_ns);
    ds4_dflash_adaptive_trace clean;
    if (!dflash_adaptive_timing(t->setup_ms) || !dflash_adaptive_timing(t->draft_ms) ||
        !dflash_adaptive_timing(t->verify_ms) || !dflash_adaptive_timing(t->heads_ms) ||
        !dflash_adaptive_timing(t->tail_ms) || !dflash_adaptive_timing(t->serial_ms)) {
        /* Actual integer cost and returned rows remain authoritative. Do not
         * admit corrupt segment timing into accumulated costs or reference. */
        a->invalid = true;
        clean = *t;
        clean.setup_ms = clean.draft_ms = clean.verify_ms = 0.0;
        clean.heads_ms = clean.tail_ms = clean.serial_ms = 0.0;
        t = &clean;
    }
    a->calls++;
    a->drafted += t->chosen;
    a->proposed += t->proposed;
    a->verified += t->verified;
    a->accepted += t->accepted;
    a->returned += rows;
    a->setup_ms += t->setup_ms;
    a->draft_ms += t->draft_ms;
    a->verify_ms += t->verify_ms;
    a->heads_ms += t->heads_ms;
    a->tail_ms += t->tail_ms;
    a->refresh_ms += t->serial_ms;
    a->zero_prefix += t->chosen != 0 && t->candidate == 0;
    a->economic_declines += t->economic;
    a->calibrations += t->calibration;
    a->serial_calibrations += t->serial_calibration;
    if (t->chosen && t->draft_ms > 0.0) {
        a->draft_ema_ms = a->draft_ema_ms > 0.0 ?
            0.8 * a->draft_ema_ms + 0.2 * t->draft_ms : t->draft_ms;
        a->draft_estimate_ms = t->draft_ms > a->draft_ema_ms ? t->draft_ms : a->draft_ema_ms;
    }
    if (t->chosen && t->funded) {
        a->funded_retries++;
        a->funded_draft_ms += t->draft_ms;
    }
    if (!rows) a->invalid = true;
    a->pending_rows = rows;
    a->pending_serial = !t->chosen && !t->verified;
    a->pending_entry = t->entry;
    a->pending_verified = t->verified;
    a->pending_accepted = t->accepted;
    a->pending_chosen = t->chosen;
    a->pending_candidate = t->candidate;
    a->pending_skipped = t->skipped;
    a->pending_economic = t->economic;
    a->pending_calibration = t->serial_calibration;
    a->pending_recovery = t->chosen && a->recovery_pending;
    if (a->pending_recovery) {
        a->recovery_pending = false;
        a->recovery_checks++;
    }
    a->pending_serial_ms = t->serial_ms;
    a->pending_reference_step_ms = a->serial_ms;
    a->pending_ms = (double)actual_ns / 1e6;
    a->pending_draft_ms = t->draft_ms;
    if (t->verified && t->verified <= DS4_DFLASH_ADAPTIVE_MAX) {
        double *ms = &a->cycle_ms[t->verified];
        const double sample = (double)actual_ns / 1e6;
        *ms = *ms > 0.0 ? 0.8 * *ms + 0.2 * sample : sample;
        const double target = t->setup_ms + t->verify_ms + t->heads_ms + t->tail_ms;
        double *cost = &a->target_ms[t->verified];
        *cost = *cost > 0.0 ? 0.8 * *cost + 0.2 * target : target;
        if (a->serial_ms > 0.0) {
            double *units = &a->target_units[t->verified];
            const double ratio = target / a->serial_ms;
            *units = *units > 0.0 ? 0.8 * *units + 0.2 * ratio : ratio;
        }
    }
    if (rows && t->serial_ms > 0.0) dflash_adaptive_serial_sample(a, t->serial_ms);
}

static inline void dflash_adaptive_ack(ds4_dflash_adaptive *a,
                                      uint32_t consumed, bool done) {
    a->reference_serial_added_ms = a->reference_verify_added_ms = 0.0;
    a->bootstrap_added_ms = a->bootstrap_reprice_ms = 0.0;
    a->reference_call_step_ms = a->pending_reference_step_ms;
    if (consumed > a->pending_rows) a->invalid = true;
    else {
        a->consumed += consumed;
        if (a->pending_serial) a->serial_consumed += consumed;
        if (a->pending_entry) a->entry_serial_tokens += consumed;
        if (consumed) {
            if (a->pending_serial_ms > 0.0) {
                /* Ordinary serial work cancels at its actual measured cost.
                 * Draft/setup/readback/report work outside that timer remains. */
                a->reference_serial_added_ms =
                    a->pending_serial_ms * consumed / a->pending_rows;
                a->reference_serial_ms += a->reference_serial_added_ms;
            } else if (a->pending_reference_step_ms > 0.0) {
                a->reference_verify_added_ms = consumed * a->pending_reference_step_ms;
                a->reference_verify_ms += a->reference_verify_added_ms;
            } else {
                a->reference_unpriced_rows += consumed;
            }
            if (a->serial_ms > 0.0 && a->reference_unpriced_rows) {
                a->bootstrap_rows += a->reference_unpriced_rows;
                a->bootstrap_step_ms = a->serial_ms;
                a->bootstrap_added_ms = a->reference_unpriced_rows * a->serial_ms;
                a->reference_verify_ms += a->bootstrap_added_ms;
                a->reference_unpriced_rows = 0;
            } else if (a->bootstrap_rows && a->serial_ms > 0.0 &&
                       a->serial_ms < a->bootstrap_step_ms) {
                a->bootstrap_reprice_ms = a->bootstrap_rows *
                    (a->bootstrap_step_ms - a->serial_ms);
                a->bootstrap_step_ms = a->serial_ms;
                a->reference_verify_ms -= a->bootstrap_reprice_ms;
                a->bootstrap_repriced_ms += a->bootstrap_reprice_ms;
            }
            a->serial_reference_ms = a->reference_serial_ms + a->reference_verify_ms;
        }
        /* Invalid diagnostics still acknowledge real consumption, but must
         * not enter arithmetic that could authorize more optional work. */
        const bool decide = !a->invalid && consumed;
        if (decide && a->pending_skipped) {
            a->skipped_steps++;
            if (a->skip_remaining && --a->skip_remaining == 0) a->reprobe = true;
        }
        if (decide && a->pending_calibration && a->serial_ms > 0.0) {
            a->need_serial = false;
            if (a->unpriced_ms < a->unpriced_rows * a->serial_ms) {
                a->engaged = true;
            } else {
                double extra = a->unpriced_ms - a->unpriced_rows * a->serial_ms;
                if (extra < a->unpriced_draft_ms) extra = a->unpriced_draft_ms;
                a->losing_cycles++;
                dflash_adaptive_backoff(a, extra, true);
                if (!done) dflash_adaptive_offer_recovery(a, a->unpriced_accepted);
            }
        }
        /* Only a fully consumed returned prefix updates the next decision.
         * EOS/cancel cannot turn unused returned rows into positive evidence. */
        if (decide && consumed == a->pending_rows && a->pending_verified) {
            const uint32_t n = a->pending_verified, accepted = a->pending_accepted;
            const double sample = (double)accepted / n;
            a->acceptance = a->observations ? 0.75 * a->acceptance + 0.25 * sample : sample;
            a->observations++;
            for (uint32_t i = 1; i <= n; i++) {
                const double prefix = accepted < i ? accepted : i;
                a->prefix_accept[i] = a->prefix_observations[i] ?
                    0.75 * a->prefix_accept[i] + 0.25 * prefix : prefix;
                a->prefix_observations[i]++;
            }
            a->next = accepted == n ? n + 1u : accepted + 1u;
            if (a->next < a->config.n_min) a->next = a->config.n_min;
            if (a->next > a->config.n_max) a->next = a->config.n_max;
        }
        if (decide && a->config.retry && consumed == a->pending_rows && a->pending_chosen) {
            const bool profitable = a->pending_accepted > 0 && a->serial_ms > 0.0 &&
                a->pending_ms < consumed * a->serial_ms;
            if (a->serial_ms <= 0.0) {
                /* Seeded first speculation remains immediate. The next
                 * ordinary target step measures serial cost before another
                 * cycle can be misclassified as profitable indefinitely. */
                a->need_serial = true;
                a->unpriced_ms = a->pending_ms;
                a->unpriced_rows = consumed;
                a->unpriced_accepted = a->pending_accepted;
                a->unpriced_draft_ms = a->pending_draft_ms;
            } else if (profitable) {
                a->engaged = true;
                a->bad_run = a->skip_remaining = 0;
                a->reprobe = false;
            } else {
                double extra = a->pending_ms - consumed * a->serial_ms;
                if (extra < a->pending_draft_ms) extra = a->pending_draft_ms;
                a->losing_cycles += a->pending_verified != 0;
                const bool preserve_interval = a->config.savings_retry &&
                    !a->pending_verified && dflash_adaptive_profitable_history(a);
                dflash_adaptive_backoff(a, extra, !preserve_interval);
                if (!done) dflash_adaptive_offer_recovery(a, a->pending_accepted);
            }
        }
        if (decide && a->config.retry && a->config.loss_meter && a->drafted && a->serial_ms > 0.0) {
            const bool recovered = consumed == a->pending_rows &&
                a->pending_verified >= a->config.n_max &&
                a->pending_accepted == a->pending_verified && a->pending_verified &&
                a->pending_ms < consumed * a->serial_ms;
            const char *floor_source;
            double trip = dflash_adaptive_loss_floor(a, &floor_source);
            const double fraction = a->config.loss_budget * ((double)a->actual_ns / 1e6);
            if (trip < fraction) trip = fraction;
            if (a->pending_recovery && !recovered) {
                a->engaged = false;
                a->skip_remaining = a->config.retry_max;
                a->reprobe = false;
            }
            if (recovered) a->probe_only = false;
            else if (dflash_adaptive_net_ms(a) > trip) {
                if (!a->probe_only) {
                    a->probe_only = true;
                    a->probe_entries++;
                    a->probe_entry_consumed = a->consumed;
                    a->skip_remaining = a->config.retry_max;
                    a->reprobe = false;
                    if (!done) dflash_adaptive_offer_recovery(a,
                        a->pending_calibration ? a->unpriced_accepted : a->pending_accepted);
                    /* A previously granted early check is still the same
                     * single opportunity if calibration also trips the meter. */
                    if (a->recovery_pending) a->skip_remaining = 0;
                }
            }
        }
    }
    a->pending_rows = a->pending_verified = a->pending_accepted = 0;
    a->pending_chosen = a->pending_candidate = 0;
    a->pending_skipped = a->pending_economic = a->pending_calibration = false;
    a->pending_recovery = false;
    a->pending_serial = a->pending_entry = false;
    a->pending_ms = a->pending_draft_ms = 0.0;
    a->pending_serial_ms = a->pending_reference_step_ms = 0.0;
    if (done) a->active = false;
}

#endif
