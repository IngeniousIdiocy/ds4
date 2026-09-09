# DFlash conservative confidence and full-block speculative

> **Scope (2026-09-09).** Conservative mode now defaults to the windowed
> cost-feedback controller described in `docs/DFLASH_GLM53.md` section 7
> (four-position minimum prefix, full-block verification, three-attempt cost
> windows with 16/32/64/128-token backoff, serial reasoning). The per-attempt
> savings ledger described below is what `DS4_DFLASH_WINDOWED=0` selects; its
> receipt fields still print in both modes, and its retry/economics gates are
> inactive under the windowed controller.

Conservative mode runs the trained seven-token draft block and selects
the longest prefix whose normalized draft probabilities meet `p_min`.
The first low or invalid confidence ends that prefix; later confident
positions cannot restart it. The target verifies only this prefix plus its
known anchor using the existing causal-prefix snapshots and rollback.
There is no fixed request-credit reserve, earning phase, or shadow-draft phase.
Later conservative retries may spend savings from already consumed work.

Public speculative is deliberately different: whenever causal conditioning
and context/response room permit, it drafts and offers all seven positions.
It does not use confidence, verifier-width economics, retry, backoff, the soft
meter, or savings funding. The target still verifies the full offered block,
and exact rollback, history ownership and the session fault latch still apply.
`DS4_DFLASH_BLIND_RESEARCH` is obsolete because `--dflash-mode speculative`
now selects this full-block path directly. The withdrawn 5% aggressive
confidence profile is no longer a public mode.

This implementation is a candidate pending end-to-end measurements. In
particular, a zero-length confident prefix still pays for a draft pass before
serial target evaluation. Confidence gating by itself does not guarantee a
throughput improvement or less than 2% loss on unfavorable input.

## Configuration

The initial candidate defaults are `DS4_DFLASH_P_MIN=0.75`,
`DS4_DFLASH_MIN_DRAFT=1`, and `DS4_DFLASH_MAX_DRAFT=7`. Probabilities must be
finite and in [0,1]; lengths must be ordered and in [1,7]. Invalid settings
disable optional work with a diagnostic. These defaults are tuning inputs,
not validated performance claims.

`DS4_DFLASH_ADAPTIVE=1` additionally adjusts the verifier prefix cap from
`DS4_DFLASH_START_DRAFT` (default 3, bounded by min/max). Full consumed-prefix
acceptance grows the cap by one; rejection shrinks it to accepted drafts plus
one. A measured unprofitable width can reduce the cap to the minimum. This is
an experimental extension; the default is `DS4_DFLASH_ADAPTIVE=0`. It never
changes the drafter's trained block geometry. These controls affect
conservative mode only; public speculative ignores them.

## Measured retry candidate

`DS4_DFLASH_RETRY=1` (conservative default) stops paying for a proposal on every serial
step after unfavorable evidence. `DS4_DFLASH_RETRY=0` reconstructs the original
confidence-prefix baseline, including its continuous checking costs. The first
conservative proposal is immediate when causal conditioning exists.
`DS4_DFLASH_MIN_SERIAL_TOKENS` accepts [0,64] and defaults to `0`; a positive
value is a diagnostic delay, not the production policy. Entry retains the causal prompt tail and captured serial rows;
only consumed ACKs increment `serial_consumed` and `entry_serial_tokens`.
`operation=entry-serial` has `entry=1`, no chosen/verified rows, and
`serial_calibration=0`. `serial_consumed_before` exposes the pre-call count.
A zero-consumption terminal ACK cannot replenish entry. New requests reset
entry; conditioning resets preserve it. If immediate entry lacks a serial
sample, the next ordinary target step measures serial cost.

A zero prefix, economic decline or measured losing attempt schedules captured
serial steps. The initial interval is the measured optional cost divided by
serial cost and `DS4_DFLASH_RETRY_TAX` (default 0.01, range [0.001,0.1]); it
can double on repeated unfavorable attempts up to `DS4_DFLASH_RETRY_MAX`
(default 512, range [1,4096]). A demonstrably useful run tolerates one isolated
miss. With savings retry enabled, a nonverified confidence/economic decline
preserves the exponent while actual accepted-prefix history still pays current
measured draft-plus-target cost. Rejection or a losing verified cycle may still
escalate it.
Only a consumed, accepted and measured-profitable cycle resets backoff.
The tax setting tunes conservative retry cadence, not a finite-request bound.
Conservative reports `profile=conservative loss_meter=1`. Public speculative
reports `policy=full-block profile=speculative`; its adaptive-named call, ACK
and totals fields are accounting only and do not gate continuous drafting.

At the verify decision, drafting is already paid. Verification economics
compare the remaining target/heads/tail cost with the value of committed rows
at the current measured serial cost. Draft cost remains charged in full and
still governs later drafting through profitability, savings and backoff. Actual target acceptance
estimates prefix yield; normalized draft confidence only selects a prefix.
Direct same-width measurements supply target costs. An unseen width uses an
explicit row estimate with a one-serial-step intercept, preferring evidence
collected after serial calibration over the first uncalibrated seeded cost. No fabricated depth
constants are used, and a cheaper serial sample never discounts measured
verifier work. Unseen widths normally calibrate only if full acceptance could
pay the estimate.

The soft loss meter is `actual_ms - reference_ms`. An acknowledged ordinary
serial/refresh row receives its own measured serial-evaluation time as reference;
setup, draft, readback and reporting outside that timer remain charged.
Verification rows are priced once at the central serial estimate recorded for
that attempt, counting only consumed rows. The central estimate is the median
of the latest up to nine successful serial samples (middle-pair average with
an even count). The observed request minimum remains diagnostic.

The initial uncalibrated block receives explicit reference on first serial
calibration. Later lower medians may reduce only this bootstrap credit; higher
medians never inflate historical credit. Conditioning reset freezes earlier
priced credit and abandons unpriced rows from the old context without erasing
their actual costs. It starts fresh local timing evidence.

The meter trips above the larger of
`DS4_DFLASH_LOSS_BUDGET * actual_ms` (default 0.03, range [0.001,0.25]) and one
full-width target cost. Receipts identify this floor as `measured`,
`row-calibrated`, `row-estimate`, or `unavailable`; an unavailable floor contributes zero.
Probe-only mode uses the capped interval and verifies only full-width
confidence prefixes. It exits only after a fully consumed, fully accepted and
profitable full-width block.

`DS4_DFLASH_EARLY_RECOVERY=1` (default; 0 disables it) permits one early
full-width recovery check per request after a measured loss with positive
committed target acceptance. Startup calibration and meter entry share this
single allowance. A full-width confidence prefix may use that check to measure
an unseen full width even when a row estimate predicts loss; known width costs
still undergo the economics check. This bounded calibration is printed. No
cold zero-prefix exception exists. An unsuccessful check schedules the capped
interval. Conditioning resets do not replenish the allowance or erase debt.

`DS4_DFLASH_SAVINGS_RETRY=1` (default; 0 reconstructs the previous retry
policy) lets completed consumed-prefix savings bypass a scheduled serial wait.
Available savings are `max(0,-net_ms)`. They must cover the larger of the latest
actual draft cost and a draft-cost EMA (0.8 previous, 0.2 new sample), after
deducting currently paid setup. This buys a next-step proposal; it never
bypasses the width economics check. The actual attempt and subsequent reporting
reduce the same ledger, with no second credit pool and no future borrowing.
When the bank is insufficient, the measured tax interval applies again.
Conditioning reset clears local draft-cost/acceptance evidence, not the ledger.

`net_percent` is a statistical scheduling estimate, not measured slowdown
against a paired serial control. Ordinary serial jitter cancels at its own
measured time; verification credit still uses an observed estimate. Use the
complete paired frontend timers for throughput claims.

The conservative scheduler uses an experimental statistical counterfactual.
Initial proposal, capture, bulk history ingestion, EOS and reporting costs can
exceed its tuning percentages on short requests. Controlled middle fixtures
where conservative was unfavorable measured about 0.3% to 0.9% slower, while
the current `003605Z` restored-prefix 24-token screen paid one 20.003 ms draft
with no verify and was 2.682% slower. These are fixture results, not bounds. Complete-request paired
timings, including startup, remain the performance evidence. Full-block
speculative has no admission limiter and can be much slower on bad input.

## Conditioning and recovery

Both DFlash modes retain completed prompt tap features through ordinary or
expert-bank prefill. A usable prompt tail directly conditions the first proposal;
it no longer requires a throwaway seed proposal. `DS4_DFLASH_CTX_CAP` defaults
to 256 retained rows, with a validated range of 1..2047 for this checkpoint.
Invalid capacity values warn and fall back to the default. If no seed is available, one exact
serial evaluation captures the first feature row. Subsequent captures contain
only the committed target prefix. Retry skips append actual target features to
a bounded circular history and linearize it only when drafting resumes. If
retention drops un-ingested rows, old draft KV is invalidated before the new
contiguous tail is ingested. No missing positions are spliced. Both pending
feature rows and the draft KV cache carry absolute frontiers. Gaps, rewinds and changed session generations
invalidate old conditioning and acceptance evidence.

Verifier snapshots remain lazy until a nonempty prefix needs verification.
A proposer failure invalidates its own cache and permanently disables DFlash
for that allocated session, in both conservative and speculative modes. After
a successful GPU drain, the failed token and every subsequent token take the
ordinary serial target path without tap capture or another proposal. A failed
drain invalidates the checkpoint and returns a request error instead. New
requests, conditioning resets, rewinds and payload restores preserve this
latch; explicitly recreating the session is the reset policy. Independently
allocated sessions remain eligible. Failures after target mutation retain the
existing complete-state recovery/error contract.

## Accounting and evidence

`decode_begin` starts a new ledger. Each returned prefix remains pending until
the CLI/server acknowledges the consumed prefix. EOS and cancellation keep all
elapsed attempt cost and count only consumed rows. An incomplete returned
block does not update acceptance evidence. Missing/invalid ACKs disable
optional work. Conditioning reset preserves request totals; it cannot hide
earlier failed work. Positive-temperature GLM stays on its ordinary path.

With `DS4_DFLASH_STATS`, each call reports `chosen` (requested model draft
positions), `proposed` (the proposer's usable output before the runtime gate),
`candidate` (the confident prefix before economics),
`verified` (the prefix actually verified), `accepted` (matching target drafts), and
`returned` (committed rows including the anchor). A completed error has zero
returned rows. Confidence values are -1 when unavailable. `bootstrap_rows`
records actual prompt-tail conditioning, not an inferred warm state.

Per-call `total_ms` includes setup, proposal, target work, heads, readback,
rollback and failure work. Segment counters report their measured portions;
on early errors the total remains authoritative. Request totals add begin,
call-report and ACK bookkeeping. `drafted` sums `chosen`, while `proposed`
sums usable proposer output, so failed or zero-confidence draft work remains
visible. `consumed` excludes unused returned rows. The final totals line
cannot include its own output cost. Complete frontend timers, including
prefill and all reporting, are the authoritative throughput comparison.

Retry receipts add `skipped`, `economic`, `calibration`, `serial_calibration`,
and `recovery` flags. ACKs expose remaining skips, meter net/reference/floor,
and the spent/pending recovery state. Totals count zero prefixes, skipped
consumed steps, declined verifies, losing cycles and each calibration type.
A separate terminal zero ACK cannot repeat the previous decision or add/reprice
reference time. `reference_source=serial-actual+verify-median9` identifies the
current ledger; earlier `min-serial` receipts use a different reference policy.

ACKs expose `reference_serial_added_ms`, `reference_verify_added_ms`,
`bootstrap_added_ms` and `bootstrap_reprice_ms`, so the change in reference is
independently reconstructible as the first three minus the fourth.
`reference_call_step_ms` records the fixed price for a verification attempt;
`serial_min_ms` and `serial_central_ms` show observed timing. Cumulative
`reference_ms = reference_serial_ms + reference_verify_ms`; the latter includes
bootstrap credit after any corrections. `bootstrap_repriced_ms` counts total
downward correction, while `reference_unpriced_rows` records the current
conditioning prefix awaiting calibration. All actual costs remain in totals.

Savings receipts add `funded`, `funding_bank_ms` and `funding_cost_ms` per call.
A funded call has a real proposer attempt and a preexisting wait; its printed
bank covers the printed measured estimate. `funded_retries` counts these calls,
and `funded_draft_ms` sums their actual draft time, including a zero confident
prefix. ACK/totals also report `savings_ms`, `draft_estimate_ms`, and
`nonescalating_declines`. No unconsumed returned row may increase savings.

Conservative policy booleans (`ADAPTIVE`, `RETRY`, `EARLY_RECOVERY`,
`SAVINGS_RETRY`, each with the `DS4_DFLASH_` prefix) validate `0` and `1` and
are ignored by full-block speculative. The same applies to conservative
`P_MIN`, `MIN_DRAFT`, `MAX_DRAFT`, `START_DRAFT`, `RETRY_MAX`, `RETRY_TAX`,
`LOSS_BUDGET`, `MIN_SERIAL_TOKENS`, and the proposer diagnostic `MIN_MARGIN`.
Public speculative also avoids the optional margin scan. In contrast, legacy
engineering switches such as `DS4_DFLASH_DISABLE`, `DS4_DFLASH_STATS` and
`DS4_DFLASH_FORCE_REPLAY` use presence: even `=0` sets them. Unset those controls
to disable them. Explicit `--dflash-mode serial` overrides legacy mode switches
and loads no drafter. `RETRY=0` is a confidence-policy baseline, not serial;
`SAVINGS_RETRY=0` isolates the latest conservative retry extension. Full controls and
precedence are in [`docs/DFLASH_GLM53.md`](../docs/DFLASH_GLM53.md).

CPU checks: `make dflash-clock-test dflash-entry-test dflash-retry-test dflash-adaptive-test dflash-confidence-test
dflash-prefix-test dflash-lifecycle-test dflash-rollback-test
dflash-sampling-test dflash-mode-test dflash-fastpath-test`. GPU evidence must
separately establish causal correctness and end-to-end behavior on SQL-8192,
JSON-512, unfavorable prose, cached300k/24 and /2048, short EOS/stop/cancel and
recovery. The long cached test must count every later retry and bulk ingest;
it cannot extrapolate only the first draft's amortized cost.
Existing request-credit receipts do not certify this candidate.

## Confidence partition CPU fixture

On Apple Metal builds, normalized confidence retains the scalar finite/max/tie
scan and ordered double partition sum, while Accelerate performs the
max-shift and per-element exponential in a reusable 154,880-float host scratch
buffer. Allocation failure uses the scalar implementation. Set
`DS4_DFLASH_CONFIDENCE_SCALAR=1` before process start to force that scalar path
for paired validation. Both paths evaluate the full vocabulary.

`make dflash-confidence-bench` builds and runs the checked-in seven-row,
154,880-vocabulary CPU fixture. On the development M3 Ultra with the normal
`-O3 -ffast-math -mcpu=native` flags, warm runs on 2026-09-06 measured the
finite/max scan at 0.514-0.533 ms per block, the scalar partition at
1.436-1.491 ms, and the Accelerate partition at 0.590-0.594 ms. The fixture's
maximum partition relative difference was 1.43e-9 and its maximum normalized
probability absolute difference was 2.20e-13; uniform and extreme rows matched
exactly. These numbers predict roughly 0.87 ms saved within a reported 24 ms
proposal block. They do not explain most proposal cost or establish an
end-to-end throughput gain.

## Drafter failure hook

`DS4_DFLASH_FAIL=drafter_once` injects one failure per process immediately
after the first successful cache-producing drafter layer. The proposer drains
the commands and uses its ordinary negative failure return. This exercises
partial-cache cleanup, the session latch and target-only continuation. The
hook cannot trigger again in later requests or independently allocated
sessions. It does not change the target verifier or target logits.

`dflash fault` receipts report the latch, mode, conditioning generation,
request number and lifetime attempt/failure/skip counters. With
`DS4_DFLASH_STATS`, `dflash request event=begin|end` markers identify request
boundaries using the generation/request pair, including speculative mode.
CPU state checks run with `make dflash-fault-test`. A live server check must
still compare a faulted response and a reused-session response with fresh
serial controls, prove constant proposer attempts after the latch, and prove
a new independent session is eligible. Run that check for both DFlash modes;
the CPU test alone is not server recovery evidence.
