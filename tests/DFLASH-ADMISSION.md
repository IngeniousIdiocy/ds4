# DFlash request-credit admission

The default policy targets less than 2% added generated-work elapsed time
(about 1.96% throughput loss) on the calibrated profile below. It is a measured
production requirement, not an unconditional hardware-time guarantee. Optional
operations can overrun their empirical estimate; such an overrun is reported
and parks speculation for the remainder of the request. Time already spent is
never erased. Acceptance/quality and causal-prefix correctness remain separate.

Presence-based switches are enabled whenever the environment variable exists,
including values `0` or an empty string. Use `env -u NAME` or `unset NAME` to
disable NO_ADAPTIVE, head, FC, script, replay and other presence switches.
BUDGET_MS is the exception here: it is parsed as a positive numeric estimate.

`DS4_DFLASH_NO_ADAPTIVE=1` selects the explicitly **uncapped experiment**. It
retains the original always-speculative path and prefill feature seeding. The
47.4t/s JSON result at commit 2ce5473 belongs to the previous policy; it must not
be attributed to this new capped default. Head/FC experiments should be screened
with NO_ADAPTIVE explicitly set, then with the intended serving policy.

## Accounting and admission

All arithmetic uses bounded integer nanoseconds. Let B be credited serial
work, A be actual timed work, and C=B+floor(B/50)-A. The next optional bundle
reserves its entire estimated cost U from existing C. Requested future tokens
and predicted acceptance never fund that reservation.

Plain serial evaluator samples supply B and the current request's minimum
observed serial-token cost. The outer call, setup, refresh, proposal, head,
readback, rollback, and failures supply A. Request-begin and acknowledgment
bookkeeping is charged too. A returned block earns nothing until the caller
acknowledges its consumed prefix. Speculative consumed rows are priced at the
minimum observed plain serial cost; a later cheaper sample reprices all such
credit downward. This reference is conservative empirically, not a proven
lower bound on an unobservable counterfactual serial run. Invalid clocks,
NaN/infinite/zero estimates, arithmetic overflow, bad acknowledgments and
missing acknowledgments fail closed.

A funded feature refresh retains the unspent reservation in escrow for its
first proposal. Its acknowledgment does not release that escrow unless the
request ends. The next proposal uses that existing reservation, avoiding a
refresh-only starvation loop. A second refresh without a proposal parks.
Success releases unused escrow; failure/cancellation keeps all actual cost
charged. A completed losing proposal reduces credit immediately, so another
proposal requires funding again. An estimate overrun parks even if acknowledged
tokens would make the whole request look profitable.

Admission precedes the roughly 1.3GiB speculative scratch, context-cache setup,
feature capture and ingestion. Capped prefill does not seed draft features;
the first funded bundle refreshes them during decode. No probe or seed work is
hidden in prefill or performed just to measure a cost. Plain serial steps leave
speculative pending features empty. New request begin clears conditioning and
all budget credit, even for a reused session with a prefix extension. A changed
session generation without begin refuses speculation and retains charged debt.

## Empirical profile v1

Authority: root-owned receipts ASTRAL-DFLASH-JSON-20260906T173129Z and
ASTRAL-DFLASH-CACHE-20260906T173358Z under
`/Users/mark/megakernel-refs/public-artifact`, based on prefix commit 2ce5473.
Largest observed shortened operation: about 266ms; observed refresh:43.5ms;
seed ingestion:22.4ms; cached-request time not assigned to these segments:
about1.5ms. Sum about333ms, with a1.5x margin, rounded to **500ms**. The same
conservative estimate initially prices cold bundles and later proposals.
It is an empirical applicability envelope, not a mathematical upper bound.

Default applicability guards:

- Apple M3 Ultra with at least500GiB physical memory; Metal already initialized.
- GLM5.3,45 trunk layers,4096 hidden dimensions,154880 vocabulary, Q8 output.
- Target file size185299232064 bytes; DFlash2 file size2342595168 bytes.
- Five-layer BF16-FC DFlash2:4096 hidden,12288 FFN,32 query/8 KV heads of128,
  five targets, block8, window2048, FC input20480, non-classic architecture.
- Configured context at most320000 (covers the302048 frontier), draft context
  at most256 rows, and seven/eight-row verifier shapes with room in live caches.
- Replay/script/head-comparison/head-benchmark/forced-draft/scalar-drafter
  diagnostics are not automatically priced by this profile.

File sizes and geometry are applicability guards, **not content identity**.
The provenance is the pinned public Q4_K-9ab7053 target and the personally
recreated DFlash2 file used by those receipts; neither file is rehashed at
request startup. Arbitrary kernel/configuration changes invalidate performance
claims until measured again. Unsupported profiles remain serial.
`DS4_DFLASH_BUDGET_MS` is an optional positive finite diagnostic override;
an invalid override refuses speculation, and an unrealistically small override
is not evidence of compliance. Normal use requires no override.

At25ms per plain serial token, earning500ms takes approximately1000 completed
serial evaluations, plus bookkeeping. JSON512 cannot fund the first probe
(about256ms credit); the cached24 case earns about13ms and stays serial without
DFlash scratch/ingest. Long requests can fund a first bundle and then use
measured savings for subsequent proposals. There is no fixed token threshold;
slower serial work earns credit faster. EOS can arrive at any time without
creating a debt against ungenerated future tokens.

## Caller contract and source scope

Call `ds4_session_decode_begin` once after sync/restore and before the first
decode opportunity. After each speculative API result, call
`ds4_session_decode_ack(session, consumed, done)`. Consumed includes a stop-string
trigger token if the ordinary serial frontend would evaluate it to recognize
the stop. It excludes sampled EOS tokens the frontend does not evaluate and
all unused returned suffix rows. done releases unused escrow; no new request
inherits it. Missing integration stays serial.

Both CLI loops, the single-session server loop and cached-depth harness are
wired. Server completion deltas include the stop-string trigger and exclude
unused block rows; cancellation before consumption credits zero. Recovery via
`decode_again` never resets the request ledger and stays serial once the prior
segment ended. Positive-temperature GLM remains on its existing ordinary path
and receives no invented speculative credit. Other backends' APIs are no-ops.

One-shot CLI ordinary serial skips evaluating its last printed token. Capped
DFlash blocks leave that last token for the same print-only branch, preventing
an extra evaluation from violating the short-request comparison. REPL/server
and the cached harness retain their existing evaluate-all scheduling.

Uncapped prefill seed arming now follows its last direct-return allocation
check. Failed in-place ring readback invalidates ring length/end/owner, rather
than preserving eligibility for partially mutated data. These are conservative
cleanup fixes, not a concurrency redesign: process-global captures still require
serialized GPU/session ownership.

## Checks

CPU-only: `make dflash-budget-test dflash-prefix-test dflash-rollback-test`.
The ledger test exercises future-credit refusal, refresh escrow, partial EOS,
cancellation, failed work, overrun, missing ACK, reused/independent sessions,
invalid acknowledgment, zero/NaN/infinite time and overflow. Scripted prefix
and replay harnesses explicitly select the uncapped diagnostic policy.

GPU owner: rerun the established cached300k/24 pair and short JSON pair with
NO_ADAPTIVE and BUDGET_MS unset. Require policy=request-credit, estimate_ms=500,
probes=0, no refresh/proposal logs and matching serial evaluation counts. Then
one longer high-accept request can establish actual funded-probe engagement.
Include an early EOS/stop case for caller settlement. Price NO_ADAPTIVE separately.
Use complete generated-block timers including setup; no GPU compliance result
is implied by a host build or these CPU tests.
