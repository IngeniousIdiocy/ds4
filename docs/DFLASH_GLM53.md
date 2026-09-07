# DFlash2 speculative decoding for GLM-5.3-Flash

`--dflash` is an optional speculative-decoding mode. The upstream checkpoint was
trained to predict one fixed block: the known target anchor followed by seven
draft positions. The default conservative profile runs that trained geometry,
computes a normalized full-vocabulary probability for each position, and offers
the longest prefix that passes the confidence floor and measured width economics.
It uses measured retry backoff with a 1% retry-tax setting, a soft request loss
meter, and retries funded only by savings from already consumed work.

Public speculative is the continuous full-block profile. Whenever causal
conditioning and context/response room permit, it drafts and offers all seven
positions without confidence admission, width economics, retry, backoff, or a
meter. In both modes the target verifies the offered causal rows and commits only
matching rows. Speculative can be much slower on an unfavorable input. The
conservative tuning inputs are scheduling controls, not request-level downside
caps.

The drafter is a **separate pretrained checkpoint** published by inco.ai, not
a piece of this repository and not derived from the target GGUF. It is
licensed **CC BY-NC-ND 4.0**, so no copy of it — original or converted — is
distributed here. This page is the recipe for building your own copy locally
for personal, non-commercial use.

- Model card: <https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2>
- Licence: <https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en>
- Method: <https://inco.ai/blog/dflash2/> and <https://github.com/z-lab/dflash>

Read the licence before you convert anything. The recipe below produces a
local file for your own use; it does not grant permission to share the
result.

**Status of this page.** The conversion/provenance recipe and explicitly
historical receipts below remain useful, but they do not certify either current
public profile. Final release measurements are recorded separately from this
implementation description. DFlash2 is optional, greedy-only and requires a
locally obtained drafter.

Both conservative and speculative modes can seed the drafter from completed
prompt tap features. Ordinary and expert-bank prefill retain the supported
recent tail; enabling DFlash no longer requires refusing expert-bank prefill.
A seed carries its owning session generation and absolute ending position.
Serial mode loads no drafter and arms no feature capture.

---

## 1. What was tested

| Component | Pinned value |
| --- | --- |
| Drafter repository | `incoai/GLM-5.3-Flash-DFlash2` |
| Drafter revision | `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| `model.safetensors` size | 2,342,169,800 bytes |
| `model.safetensors` sha256 | `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b` |
| Converted GGUF size | 2,342,595,168 bytes |
| Converted GGUF sha256 | `a4bbfbd9e5db62ea31c5cde0bab38a4f9be11a8005dfd078f0d455bb630d66a9` |
| Target GGUF | `GLM-5.3-Flash-Q4_K-9ab7053.gguf` |
| Target GGUF sha256 | `828f413cae8ceee74796606814295e00e04c1c7b151aca9a3c5c75db32cb73e0` |
| Converter | `gguf-tools/dflash2_to_gguf.py` (this repository) |
| Python | 3.12, numpy 2.5.1 (numpy is the only third-party import) |

### Historical capability receipt on `b723dfa`

One deliberately repetitive SQL prompt was run for a fixed 8,192-token horizon
through all three startup policies. Each arm emitted 22,036 bytes with the same
digest in that run:

| startup policy | generated tokens | generation time | tokens/s |
| --- | ---: | ---: | ---: |
| serial | 8,192 | 212.310 s | 38.5850 |
| conservative | 8,192 | 173.536 s | 47.2064 |
| speculative | 8,192 | 134.764 s | 60.7875 |

Receipt: [`dflash-three-mode.json`](../bench/receipts/glm53-m3ultra/dflash-three-mode.json),
build `b723dfa`. The compact receipt contains the exact prompt. It asks
for 2,000 SQL tuples, but every arm stops at the fixed token limit during tuple
483. This is a favorable fixed-horizon capability result, not a completed-task
measurement or a representative average. Matching bytes on this fixture do not
create a universal byte-identity requirement; meaningful quality comparisons
remain about the target distribution, argmax margins and causal output effects.

The drafter's `main` branch has moved on since that revision and its current
model card differs from the one tested here. **Pin the revision.** A different
drafter checkpoint may load and may even draft, but nothing on this page has
been measured against it.

The target GGUF matters too. Its `token_embd` and `output` tensors are Q8_0
and its experts are Q4_K; the drafter borrows both of those target tensors
(section 4), so a differently quantized target changes acceptance and
throughput even though the drafter itself is unchanged. Requantizing the
target does **not** require reconverting the drafter.

## 2. Get the drafter

Keep the weights outside the repository, or in a directory git ignores —
`dflash2-weights/` at the repository root is ignored for exactly this. The
commands below use `$DFLASH_DIR`; point it anywhere you like.

```sh
export DFLASH_DIR="$HOME/models/dflash2-glm53"
pip install "huggingface_hub>=0.34" numpy

hf download incoai/GLM-5.3-Flash-DFlash2 \
    --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 \
    --local-dir "$DFLASH_DIR/model"
```

On `huggingface_hub` older than 0.34 the command is spelled
`huggingface-cli download` with the same arguments. Only `config.json` and
`model.safetensors` are needed; the rest of the repository is small.

Verify what you got before converting it:

```sh
shasum -a 256 "$DFLASH_DIR/model/model.safetensors"
# b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b
```

If that hash differs, stop: you have a different checkpoint from the one this
page describes.

## 3. Convert to GGUF

```sh
python3 gguf-tools/dflash2_to_gguf.py "$DFLASH_DIR/model" "$DFLASH_DIR/GLM-5.3-Flash-DFlash2.gguf"
```

The converter reads only the drafter directory: `config.json` for the
architecture and `model.safetensors` for the weights. It does not read the
target GGUF and bakes nothing about it into the output. BF16 bulk weights pass
through unchanged; small norm vectors and convolution bases are widened
losslessly to F32. The conversion is a deterministic byte transform, so the
output is reproducible:

```sh
ls -l "$DFLASH_DIR/GLM-5.3-Flash-DFlash2.gguf"     # 2342595168 bytes
shasum -a 256 "$DFLASH_DIR/GLM-5.3-Flash-DFlash2.gguf"
# a4bbfbd9e5db62ea31c5cde0bab38a4f9be11a8005dfd078f0d455bb630d66a9
```

Both values above were confirmed by re-running exactly this command on the
pinned source: same size, same digest, byte for byte.

The resulting file carries: hidden size 4096, vocabulary 154880, five layers,
block size 8, sliding window 2048, rope base 10000, mask token 154856, and
tap layers `[5, 14, 24, 33, 42]`.

## 4. What the drafter borrows from the target

The drafter is not self-contained. On every proposal it reads the **anchor
row** (the token just committed) and the **mask row** from the *target's*
`token_embd`, and it runs its own block head through the *target's* `output`
weight. It is also fed hidden states captured from the target at its five tap
layers.

This is why the target's quantization is part of the configuration and why the
engine screens the target before setting anything up. At load time it checks
that

- `token_embd` and `output` are both 2-D and agree on width and row count,
- the width matches the drafter's hidden size (4096),
- `token_embd` has a row decoder for its type — F32, BF16, Q8_0, Q4_K, Q5_K or
  Q6_K,
- the drafter's mask token (154856) exists in the target's table, and
- every tap layer index is inside the target's layer count.

A target that fails any of these is refused at load with the reason on stderr,
and the engine continues without speculation. It never aborts mid-generation.

## 5. Run it

CLI:

```sh
./ds4 --model /path/to/GLM-5.3-Flash-Q4_K-9ab7053.gguf \
      --dflash "$DFLASH_DIR/GLM-5.3-Flash-DFlash2.gguf" \
      --dflash-mode conservative \
      --ctx 65536 --temp 0 -p "your prompt"
```

Server: the same `--dflash FILE` flag. Set `temperature: 0` on each API
request intended to exercise DFlash; positive-temperature requests stay serial.

```sh
./ds4-server --model /path/to/GLM-5.3-Flash-Q4_K-9ab7053.gguf \
             --dflash "$DFLASH_DIR/GLM-5.3-Flash-DFlash2.gguf" \
             --dflash-mode conservative \
             --ctx 65536
```

`--dflash-mode` is a startup policy shared by the CLI and server:

- With neither `--dflash` nor an explicit mode, startup is serial and allocates
  no drafter state.
- `conservative` uses the confidence/retry policy described in section 7. This
  is the default when `--dflash FILE` is supplied without an explicit mode.
- `speculative` continuously drafts and offers the full seven-position block
  whenever causal conditioning and context/response room permit. It does not
  use confidence, width economics, retry, backoff, or a loss meter. It can be
  much slower on unfavorable input.
- `serial` does not load the draft model or allocate DFlash scratch, even when
  a `--dflash FILE` path is present.

Speculative and conservative modes require `--dflash FILE`; without draft
weights they fail at startup. Every speculative draft must agree with the
target verifier before it is accepted. Batched target arithmetic can differ
from ordinary serial arithmetic, so either DFlash mode can produce a different
continuation. Byte equality and quality need their own evidence; a confidence
threshold is not a quality certificate. Only serial mode follows the ordinary
serial numerical path. There is no hard overhead cap: even one paid draft can
be material on a short request.

An explicit `--dflash-mode` takes precedence over the legacy
`DS4_DFLASH_DISABLE` and `DS4_DFLASH_NO_ADAPTIVE` presence switches regardless
of argument order. When the flag is omitted, those switches retain their
legacy meanings.

A successful load prints one line naming the drafter's geometry and the
target's borrowed tensor types, for example:

```
ds4: dflash bind rope_base=10000.0 rms_eps=1e-05 window=2048 mask=154856
ds4: DFlash2 draft model loaded: .../GLM-5.3-Flash-DFlash2.gguf (layers=5 block=8 targets=5 window=2048 target_embd_type=8 target_head_type=8 vocab=154880)
```

Type `8` is Q8_0. `mask=154856` must match the drafter's config; a `mask=0`
here means the drafter did not bind its metadata.

### Controls

| Variable | Effect |
| --- | --- |
| `DS4_DFLASH_DISABLE=1` | With no explicit `--dflash-mode`, select serial startup and do not load the drafter. |
| `DS4_DFLASH_NO_WIDE_ROLLBACK=1` | Kill switch. Refuse the speculative cycle on any graph whose speculative state includes the DSA indexer tail — that is, every real GLM-5.3 graph — before any target state is mutated, and decode serially. This is the behaviour of the branch before the rollback was completed. |
| `DS4_DFLASH_FORCE_REPLAY=1` | Take the restore-and-replay rollback even where the cheaper per-step snapshot would be sound. An A/B oracle, not a normal setting. |
| `DS4_DFLASH_NO_ADAPTIVE=1` | With no explicit `--dflash-mode`, select public full-block speculative mode. A mode flag still takes precedence. |
| `DS4_DFLASH_STATS=1` | Per-call stages, conservative confidence decisions, request begin/consumed-ACK/end markers and public-profile request totals on stderr; also retains verifier and engine diagnostics. |
| `DS4_DFLASH_ZERO_FEATURES=1` | Zero the drafter's input features. It makes a mismatch *likely*, not certain: the drafter still emits a deterministic token per row, and a common one occasionally coincides with the target's own prediction. |
| `DS4_DFLASH_FORCE_DRAFTS=N` | Diagnostic cap on the offered/verified draft prefix. Conservative mode still runs the trained seven-position proposal; this is not a way to measure a cheaper N-position drafter. It does not force a rejection location. |
| `DS4_DFLASH_MIN_MARGIN=N` | Optional top1-minus-top2 proposer-margin diagnostic for conservative and legacy/golden calls. Public full-block speculative ignores it and avoids its optional margin scan. |
| `DS4_DFLASH_CTX_CAP=N` | Retained target-feature/draft-context rows; default 256, validated range 1..2047 for this checkpoint. Invalid values warn and use the default. Larger histories change capture/ingestion cost. |
| `DS4_DFLASH_SCRIPT=P:K:N,...` | Deterministic draft supply (test-only): at absolute position P propose N drafts of which the first K are the retained serial continuation, with a known-wrong token at row K. The verifier is untouched and decides for itself. Unnamed positions decode serially, so blocks land exactly where asked. See `tests/dflash_rejection_harness.sh`. |
| `DS4_DFLASH_SCRIPT_IDS=FILE` | The retained continuation the script proposes from: a base position followed by one token id per line. |
| `DS4_DFLASH_SCRIPT_SERIAL=1` | Control arm: same binary, same dumps, every position serial. |
| `DS4_DFLASH_SCRIPT_DUMP=DIR` | At each scripted frontier, write the frontier logits and a per-tensor digest of the complete speculative state. |
| `DS4_DFLASH_FAIL=point[,...]` | Failure diagnostics: `state_save`, `after_arm`, `after_verify`, or `drafter_once`. The last fails one cache-producing drafter layer per process and exercises the persistent session fallback described below. |

Conservative-policy controls use explicit values; their boolean settings must
be `0` or `1`. Invalid conservative settings disable optional work with a
diagnostic. Full-block speculative ignores these controls.

| Variable | Default | Effect of the control |
| --- | ---: | --- |
| `DS4_DFLASH_P_MIN` | 0.75 | Finite normalized draft probability in [0,1]; first low/invalid position ends the prefix. `0` retains every finite position before other gates. |
| `DS4_DFLASH_MIN_DRAFT` / `DS4_DFLASH_MAX_DRAFT` | 1 / 7 | Verifier prefix range, ordered within [1,7]. Neither changes the trained drafter block. |
| `DS4_DFLASH_ADAPTIVE` | 0 | `1` enables experimental accepted-count verifier-cap adaptation; `0` fixes the cap at max. |
| `DS4_DFLASH_START_DRAFT` | 3 | Initial adaptive cap, bounded by min/max; ignored with adaptation off. |
| `DS4_DFLASH_MIN_SERIAL_TOKENS` | 0 | Diagnostic delay before the first proposal, counted only from actually consumed pure-serial tokens, range [0,64]. `3` exercises the withdrawn three-token startup policy; it is not the public default. |
| `DS4_DFLASH_BLIND_RESEARCH` | obsolete | No effect. Full-block behavior is selected directly with `--dflash-mode speculative`. |
| `DS4_DFLASH_RETRY` | 1 | `0` disables retry, economics, calibration and soft-meter scheduling, reconstructing the continuous confidence-prefix baseline. It does not select serial mode. |
| `DS4_DFLASH_RETRY_MAX` | 512 | Maximum skipped serial steps, range [1,4096]. |
| `DS4_DFLASH_RETRY_TAX` | 0.01 | Conservative measured retry cadence tuning fraction, range [0.001,0.1]; not a guaranteed slowdown. |
| `DS4_DFLASH_LOSS_BUDGET` | 0.03 | Conservative-only soft loss-meter fraction, range [0.001,0.25]; `0` is invalid, not an off switch. |
| `DS4_DFLASH_EARLY_RECOVERY` | 1 | `0` removes the single early full-width recovery allowance. |
| `DS4_DFLASH_SAVINGS_RETRY` | 1 | In conservative mode, `0` disables savings-funded wait bypass and the associated non-escalating decline rule, reconstructing the preceding retry policy. |

Legacy boolean engineering switches in the first table use **presence**, including an
empty value or `0`; unset them to turn them off. Explicit `--dflash-mode serial`
is the reliable startup kill. The mode flag overrides the legacy mode switches.
The policy booleans in the second table instead honor `=0` as documented.

Existing kernel controls include `DS4_DFLASH_DISABLE_FC_MM` and
`DS4_DFLASH_DISABLE_HEAD_NT4` (kill), paired with `DS4_DFLASH_FC_MM` and
`DS4_DFLASH_HEAD_NT4` (force eligible shapes). Kills win over forces, and forces
do not relax tensor-shape checks. `DS4_DFLASH_FORCE_REPLAY` is the target-state
oracle. These are presence controls: `=0` still sets them. Detailed kernel
experiments remain in `docs/GLM53_M3ULTRA.md`; keep them out of policy timing
comparisons unless that experiment is the declared variable.

The serial-capture epilogue candidate, verifier BF16-MM candidate, verifier
indexer candidate and padded proposer-head candidate are experimental and off
by default. Their force/comparison controls do not define either public profile.
They remain disabled for release measurements unless the receipt explicitly
names the experiment; none should be inferred from `conservative` or
`speculative` mode alone.

## 6. How rejection is undone

A verify forward advances more per-token state than the tokens it commits. On
GLM-5.3 that state is the KDA convolution and recurrent buffers **and**, on
graphs with full DSA layers, the indexer tail K+gate ring. If part of a
drafted block is rejected, all of that has to go back to the frontier the
committed prefix reached — otherwise the rejected rows stay in the ring while
the checkpoint advances, and the next token is computed against state that no
serial run would ever have produced.

Before target verification, the runtime preserves a complete pre-block backup
when the graph requires it. During verification it captures the complete KDA
and DSA-tail state at each causal prefix. Partial acceptance restores the
snapshot for the committed prefix; full acceptance needs no rollback. Missing
or incomplete prefix capture falls back to the backup and serial replay.
`DS4_DFLASH_FORCE_REPLAY=1` forces that oracle even for a fully accepted block.
Rejected future compact-cache entries remain invisible until overwritten;
dense-cache validity advances only over committed rows.

All restore/replay work is included in call timing. The default retains the
batched target arithmetic of accepted rows; restoring a sound causal prefix
does not imply byte equality with a separately evaluated serial run. See
[`tests/DFLASH-PREFIX.md`](../tests/DFLASH-PREFIX.md) for the accepted-prefix and
changed-rejected-suffix checks.

A negative drafter result occurs before target mutation. After a successful
GPU drain it disables DFlash for that allocated session, and the current token
and later calls use ordinary serial evaluation without draft/capture retries.
Requests, rewinds, payload restores and conditioning resets do not clear this
latch. Recreating the session is the reset; an independent session remains
eligible. If the GPU drain is unsafe, the request fails with an invalid
checkpoint instead of reusing uncertain state. Failures after target mutation
require full state recovery or request invalidation.

## 7. Conservative admission and full-block speculative

The drafter keeps its trained anchor-plus-seven block. Conservative takes the
longest prefix whose normalized per-position probability meets `p_min`; it
stops at the first low or invalid position. Verification uses that prefix plus
the known anchor. A zero prefix still pays the drafter before one ordinary
serial evaluation. Normalized confidence is not assumed to be the probability
that the target will accept a token.

Speculative instead offers all seven positions on every eligible step. It does
not consult normalized confidence, width economics, the retry schedule, the
soft meter, or the savings ledger to decide whether to draft or verify. Missing
prompt conditioning still requires the normal cold refresh, and context or
response capacity can clamp the block. Its `adaptive`-named call, ACK and totals
records remain accounting records; they do not make the full-block profile
adaptive.

Neither public profile has a deliberate serial-token delay before its first
eligible proposal. A tiny answer can pay
one draft and a cold verifier without enough tokens to amortize their cost;
this is a disclosed limitation, not a latency design target.
`DS4_DFLASH_MIN_SERIAL_TOKENS` is an off-by-default diagnostic (`0`). Setting it
to `3` defers the first conservative proposal until three consumed serial ACKs
while retaining the causal prompt tail and serial features. Conditioning resets
retain that conservative request counter; a new request resets it. Full-block
speculative ignores the setting. If immediate conservative entry lacks a serial
sample, one actual serial step calibrates cost. There is no fixed earning period,
500 ms reserve, throwaway seed proposal or shadow phase.

In conservative mode, after a zero prefix, economic decline or measured losing cycle, the runtime
schedules serial steps using measured optional cost divided by serial cost and
the retry-tax setting, doubling repeated unfavorable attempts up to the cap.
A useful run tolerates one isolated miss. A fully consumed, accepted and
profitable cycle resets backoff. A confidence/economic decline that never
consulted the verifier does not increase the geometric exponent while actual
accepted-prefix history still pays current measured cost; rejection or a
losing verified cycle may escalate it.

At the verify decision the draft is already paid. The width guard compares
remaining target/heads/tail cost with expected committed-prefix value, using
prior target acceptance. The draft remains charged in full and governs whether
to draft again through profitability, savings and backoff. Known widths supply
measured cost; unseen widths use a labelled row estimate with one serial-step
intercept, preferring serial-calibrated evidence once available. A direct
same-width measurement remains authoritative, even if it was the first cold
cycle. A cheaper serial sample does not discount measured verifier work.

The consumed-prefix ledger is `actual_ms - reference_ms`. An acknowledged ordinary
serial/refresh row is credited at its own measured serial-evaluation time;
setup, drafting, readback and reporting outside that timer remain actual cost.
Consumed verification rows receive a fixed price from the central serial
estimate recorded for that attempt. This estimate is the median of the latest
up to nine successful serial samples, averaging the middle pair for an even
count; the observed minimum is printed separately.

The first uncalibrated seeded block receives explicit credit on its first
serial calibration. Later lower medians can reduce that bootstrap credit;
later higher estimates never reprice historical rows upward. Conditioning
reset freezes earlier priced reference and drops old uncalibrated reference
eligibility while retaining every actual cost. Thus ordinary serial jitter
cannot manufacture savings or debt, and no unconsumed suffix earns credit.
In conservative mode, the soft meter enters
probe-only mode above the larger of `loss_budget * actual_ms` and one full-width
target cost, measured or explicitly estimated from another width. An unavailable
floor contributes zero. Probe-only uses the capped interval and full-width
confidence prefixes; only a fully consumed, fully accepted and profitable
full-width block exits it.

One early full-width check per request is available after positive committed
target acceptance followed by measured loss. Startup calibration and meter
entry share this single allowance. It may measure an unseen full width despite
a pessimistic row estimate, but known width economics still apply. An
unsuccessful check returns to the capped wait. Cold zero prefixes get no
exception, and conditioning resets cannot replenish it.

In conservative mode, completed savings `max(0,-net_ms)` may fund the
next draft instead of a scheduled serial step. They must cover the larger of
the latest draft cost and its EMA, after deducting currently paid setup. Every
new attempt reduces the actual ledger; no future output or unconsumed suffix
can fund it. A depleted bank returns to the measured retry interval. Funding
a proposal never bypasses target-prefix economics.

Full-block speculative retains actual-cost and consumed-prefix accounting for
diagnostics and correct lifecycle ownership. Those records do not gate later
drafts.

Serial skips retain actual tap features in a bounded session ring. Resuming
linearizes that contiguous tail once; if retention dropped un-ingested rows,
the old draft KV is invalidated before ingest. Session ownership and absolute
positions prevent gaps or another request's features being spliced in.

**Greedy only.** `--dflash` speculates for greedy decoding. A request with a
positive temperature decodes serially instead, and this is deliberate: the
sampled cycle signals a rejection by masking the rejected token's logit, and
every caller then re-applies top-k/top-p/min-p to that modified vector — which
filters the residual against a different support than the target's own
distribution and can admit a token the original filter excluded.
`tests/test_dflash_sampling.c` demonstrates it on the real sampler. Fixing it
means computing the residual against the original filtered support; until
that exists, positive temperature does not enter the cycle.

Per-call timing includes allocation/setup, draft and bulk context ingestion,
verification, heads, readback, restore/replay, serial fallback and errors.
Request totals add decode-begin, call-report and ACK bookkeeping. EOS, stop
and cancellation retain all paid attempt cost and credit only the consumed
prefix; a later terminal zero ACK cannot repeat the previous decision.
Conditioning resets clear local evidence but preserve request costs, consumed
counts, pending retry policy and the spent recovery allowance.

The final totals line cannot time its own output. Prompt capture occurs during
prefill; model loading is startup work. Complete frontend prefill/generation
and wall timers are the authoritative paired comparison, with matching prompt,
context and stopping policy. Receipts print exact per-ACK reference additions
and bootstrap corrections, plus `reference_source=serial-actual+verify-median9`. The internal reference ledger is a scheduling
estimate, **not measured slowdown versus a serial control**. Short requests can
pay material initial cost even when a longer request amortizes it. In controlled
middle fixtures where conservative was unfavorable, paired complete-generation
results were about 0.3% to 0.9% slower, but this is measured fixture evidence
rather than a bound. The current `003605Z` restored-prefix 24-token screen paid
one 20.003 ms proposal with no verification and was 2.682% slower than its serial
control.
Neither the conservative 1% retry setting nor its 3% soft meter is a hard
request-level cap. Full-block speculative has no such limiter and can be much
slower on bad input.

See [`tests/DFLASH-CONFIDENCE.md`](../tests/DFLASH-CONFIDENCE.md) for receipt
fields and focused checks. Existing request-credit receipts do not certify
this policy. Drafter-kernel numerical changes are judged by proposal decisions,
acceptance and measured request benefit; target-kernel changes retain their
separate causal and numerical validation contract.

## 8. Smoke test

Confirm the policy and all paid attempts are visible:

```sh
DS4_DFLASH_STATS=1 ./ds4 --model TARGET.gguf --dflash "$DFLASH_DIR/GLM-5.3-Flash-DFlash2.gguf" \
    --dflash-mode conservative --ctx 8192 --temp 0 -n 64 \
    -p "Write a short paragraph about the sea."
```

Expect `dflash admission policy=confidence-prefix` with the resolved controls,
`dflash adaptive` calls, consumed `adaptive_ack` lines, and a terminal
`adaptive_totals done=1`. A nonzero `chosen` proves a paid proposer attempt;
`verified=0` can be the correct confidence/economics decision. It is not
necessary for an unfavorable prompt to produce an accepted draft. Totals
must retain such calls and their draft cost. `funded=1` identifies a retry
bought with completed savings, with its bank and cost estimate printed.

Compare with `--dflash-mode serial` for end-to-end performance. Use
`DS4_DFLASH_SAVINGS_RETRY=0` to isolate savings policy and `DS4_DFLASH_RETRY=0`
to reproduce continuous confidence checking; neither is a serial baseline.

Then the rollback check. Forcing a rejection needs the drafts themselves to be
chosen, not just capped: `DS4_DFLASH_FORCE_DRAFTS` sets the block length and
`DS4_DFLASH_ZERO_FEATURES` only blanks the drafter's inputs, so together they
exercise rejection at the *first* draft for a few block lengths and nothing
else. `tests/dflash_rejection_harness.sh` supplies the drafts instead:

```sh
DS4=./ds4 MODEL=TARGET.gguf DFLASH="$DFLASH_DIR/GLM-5.3-Flash-DFlash2.gguf" \
  sh tests/dflash_rejection_harness.sh
```

It runs one serial control pass to record the true continuation, then one
speculative pass whose blocks propose that continuation for K rows and a
known-wrong token at row K — the verifier still decides — for rejection after
0, 1, 2 and 3 accepted drafts, at start positions covering all four residues
of the 4-token pool boundary, plus a full-accept block and the shortest block
that can reject. It compares the token streams, the per-block accept and
rollback outcome, the frontier logits, and a per-tensor digest of the complete
speculative state at equal frontiers.

The rejection harness forces replay. For that control, expect output identical
to its serial fixture and `accepted` equal to each block's K;
`rollback=replay` on both partial and full-accept blocks; and the restored speculative state bit-identical to the
serial arm's at the same frontier (both arms reach those positions through
the same serial kernels, so the digests must match exactly — a difference is a
layout or ordering defect, not drift).

`DS4_DFLASH_NO_WIDE_ROLLBACK=1` reverts to serial decode with a one-line
notice; that path is unchanged from before and is the fallback if anything
here misbehaves.
