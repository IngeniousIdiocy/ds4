# Fidelity policy for the GLM-5.3-Flash Metal branch

This file states what a kernel or graph change on this branch has to prove before it
ships, and keeps the ledger that binds the branch's numerics to clean upstream. The
short version of the background research (what llama.cpp, vLLM, SGLang, FlashInfer,
Marlin/GPTQ, TensorRT-LLM, MLX and PyTorch actually accept) is that no reputable
project requires bit-exactness for throughput-motivated floating-point reordering, and
the one statistically principled gate found (TensorRT-LLM's) bounds a change by the
measurement's own noise rather than by a fixed percentage. The rules below follow that
shape, with one addition: everything that is not bit-exact can be switched off at once
(`DS4_GLM_EXACT=1`, see `bench/EXACT-MODE-PLAN.md`).

## Tier 1 — no floating-point order change

A Tier 1 change computes the same expression over the same values in the same order;
only dispatch shape, lane assignment, staging or memory traffic changes. It must show:

- **Standard gate output byte-identical** to the reference: the retained short-context
  greedy output (`-c 8192 --nothink -n 512 --temp 0` on the standard math prompt), and
  the depth gate (`-c 70000` on the 62k-token needle prompt) byte-identical between arms.
- **Kernel bit-identical to the production kernel in a randomized harness:** at least
  50,000 draws, NaN-poisoned outputs, production threadgroup geometry, every output
  word the reference writes compared word for word, zero poison survivors.
- **Determinism:** 1,000 identical-input runs, zero differing bits.
- **Harness hygiene** (below). A campaign that cannot report a finite fraction and a
  control arm detected on every planted draw is void, not merely weak.

Tier 1 changes need no exact-mode registration (there is nothing for exact mode to
restore) but keep a kill switch for A/B measurement.

## Tier 2 — floating-point order changes (never quantization)

Preconditions:

- **P1** every output the change does not touch is bit-identical over at least 50k
  poisoned draws, and the touched output's difference distribution is reported;
- **P2** determinism 1000/1000;
- **P3** the DFlash-vs-serial byte contract holds on the same build (where DFlash is
  available; see `docs/GLM53_M3ULTRA.md` for its status on this branch);
- **P4** standard and depth gate outputs compared and reported (byte identity expected
  in practice, not required).

**Scorer gate.** Build the teacher-forced scorer (`make -C gguf-tools quality-score`,
rebuilt after every source change; a stale scorer silently reports no change), run
`gguf-tools/quality-testing/score_official MODEL manifest.tsv OUT.tsv` over
`gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv`, and
compare with `gguf-tools/quality-testing/compare_scores.py` — always against the
**same-build control** (change off, same binary), using the 100 per-prompt `avg_nll`
values:

- **S1** |Δ mean avg_nll| ≤ 2 × the paired standard error of the 100 per-prompt deltas
  (measured, not assumed).
- **S2** hard ceiling regardless of SE: |Δ avg_nll| ≤ 1e-4 absolute.
- **S3** exact two-sided binomial test on per-prompt wins vs losses (ties excluded):
  p > 0.05.
- **S4** `first_match` within ±1 of 100 and `avg_lcp` within ±0.1 of the control.
- **S5** cumulative drift budget: the branch's avg_nll must stay within **3e-4
  absolute** of clean upstream antirez/ds4 at the pinned commit, scored with the same
  GGUF and the same upstream scorer on the same machine. The pin moves only by an
  explicit decision, never as a side effect of a build.

**Exact mode.** Every Tier 2 change ships behind its own kill switch *and* is registered
in the exact-mode list, so that `DS4_GLM_EXACT=1` turns every FP-order change off and
the result can be compared with the upstream pin and the same-build all-off arm. This
is a measured diagnostic, not a universal upstream byte-identity promise: corrected
behavior outside the registry may remain active. The check is the recipe in
`bench/EXACT-MODE-PLAN.md` ("The exact-mode check").

**Ship default-on** only if S1–S5 pass and the interleaved gate (≥ 4 repetitions) shows
a mean gain of at least +0.2 t/s; a kill switch is mandatory and the feature is
registered in the exact-mode list; the report carries Δ, SE, |Δ|/SE, wins/losses/ties,
p, first_match, avg_lcp and cumulative drift.

## Tier 2 for prefill-side changes

A prefill-side floating-point-order change perturbs every later token through the KDA
gates and recurrences. Two consequences were measured: on the 100-prompt manifest the
scorer's paired standard error for such a change is 5.5–6.3e-4, so the decode-era S2
ceiling (1e-4) and S4 (avg_lcp ±0.1) sit below the instrument's noise and can prove
neither harm nor safety; and end-to-end logit distance against an exact-arithmetic
reference cannot rank two tiny perturbations either — changing one small matmul to FP64
moves the final logits by about 0.4 for production and candidate alike. Greedy output
identity is therefore not a usable gate for prefill changes: main itself would fail it
against a more accurate copy of itself.

**Registration remains absolute:** every floating-point-order change ships behind its
own kill switch and is registered in `glm53_exact_mode()` so `DS4_GLM_EXACT=1` turns
it off. The upstream diagnostic reports numerical and quality differences; it does not
turn byte identity into a model-quality requirement. The standard below decides whether
such a flag may default on.

Gate for a prefill-side Tier 2 change (all required; P1–P4 as above; determinism
1000/1000 in the kernel harness *and* repeated in-graph runs byte-identical):

1. **Kernel accuracy against the exact result.** Over ≥ 50k poisoned draws at production
   geometry with a finite-dominated fill (≥ 20% finite output words) and a
   known-different control arm that must fail, compute the reference in FP64
   sequential arithmetic with the kernel's own staging semantics (the same narrowing of
   inputs main applies, so the comparison isolates ordering error) and, separately, the
   error main's own input narrowing already introduces (the "rounding floor"). Pass:
   the candidate's max and mean ordering error against the exact result are ≤ one
   tenth of that rounding floor. Main's own ordering error is reported alongside; the
   candidate need not beat it. This bounds the candidate's error to a small fraction of
   the input-narrowing error main already accepts; it does *not* follow that the model
   cannot see it, which is why criterion 2 is assessed separately and never inferred
   from this one. For a family with no input narrowing (an all-FP32 router, for
   instance) the floor is zero and the rule is vacuous; there the governing rule is
   the exact-reference gate's (`bench/README.md`, "Exact-reference gate"): the candidate's
   maximum and mean ordering error against the FP64 reference must be no worse than
   production's, with all three numbers reported.
   Never divide by a zero floor or manufacture a floor from unavailable
   pre-quantization weights.
2. **Prediction quality statistically indistinguishable from main.** The teacher-forced
   scorer on the large manifest (≥ 1,000 prompts; `bench/fidelity/manifest-1k/` — the
   100-prompt set stays as the upstream-pin drift ledger), candidate vs same-build
   control: |paired mean Δ avg_nll| ≤ 2 × the paired SE; wins/losses symmetric (exact
   binomial p > 0.05); first_match within ±1 of the control; cumulative drift vs the
   upstream pin on the 100-prompt set ≤ 3e-4 absolute (S5). S2 and S4 are reported, not
   gating. `bench/tier2-gate.sh` runs this and `bench/compare_1k.py` computes the
   verdict.
3. **Speed on the long regime:** ≥ +1% on the 62k bucket, interleaved ≥ 4 repetitions,
   every pair positive (short buckets must not regress beyond noise).
   `bench/prefill-gate.sh` is the instrument.

Ship default-on only if 1–3 pass; opt-in otherwise; kill switch and exact-mode
registration always.

### Two modes, one bundle

There is no per-feature user-facing choice. The exact-mode registry in
`glm53_exact_mode()` defines the fast bundle: `DS4_GLM_EXACT=1` means every registered
item is off; the default means every adopted item is on. The per-feature
`DS4_GLM_DISABLE_*` / `DS4_GLM_ENABLE_*` switches remain engineering
knobs for bisecting a speed or fidelity regression to one change; no adopted item ships
opt-in on its own. Adoption is judged per round: each candidate passes criterion 1 (the
check that a kernel is not wrong), then the *bundle* with every candidate on is scored
once on the 1k manifest against the same-build control (criterion 2, including the
cumulative-drift budget on the 100-prompt set) and gated once on speed (≥ +1% at 62k).
Items that fail criterion 1 are excluded individually; a bundle that fails criterion 2
is bisected with the engineering switches. Every registered item's enable switch
selects the *gated* shape by default (internal knobs — block size, slice count, tile —
default to the measured-best configuration), and the registry entry names that shape,
so enabling the bundle never selects an untested variant.

## Harness hygiene for bit-exactness campaigns

A bit-exactness campaign that only ever reports "zero mismatches" proves nothing on its
own. One early harness wrote NaN and Inf into activation buffers that were shared
across draws and never cleaned, so the poison accumulated; by mid-campaign most tokens
had a stale NaN somewhere in their activation row, every output of that token was NaN,
only 62% of the compared words were finite, and the control arm was invisible on 78 of
79 misses because doubling a NaN row changes nothing. Five rules follow:

1. **Fresh activations on every draw.** Refill the whole activation region with fresh
   finite values at the top of every draw (on the GPU — a CPU refill of a 4096-token
   buffer costs more than the matmuls it feeds), then poison the copy this draw may
   poison.
2. **Print the finite fraction, and require ≥ 20%.** "Zero mismatches over N words" is a
   certificate only when read as "of which X% were finite".
3. **A known-different control arm, detected on every draw it is planted.** Replace one
   whole token's activation row with fresh finite values: every one of that token's
   output words must move. Flipping a single mantissa bit is not enough (detected on 65
   of 120 arms in one campaign). Detection on every planted draw is the criterion; the
   fraction of output words moved is a strength report.
4. **Compare against the kernel that actually shipped.** When a change rewrites a shared
   template, the "plain" instantiation of the rewritten template is not the pre-change
   kernel. Copy the pre-image kernel verbatim out of the earlier commit, rename it, and
   compile it alongside.
5. **Prove the inputs to both arms are identical.** If a change also touched the work-map
   kernel, build the map with both the production and the pre-image map kernel and
   compare the work lists word for word.

## Known upstream defect: DSA selection tail sentinels reach a valid-only attention kernel

`kernel_glm53_expand_pool_selection` writes `0xffffffff` into the tail slots (2048–2050 of
the 2,051-wide selection) that a token's visible length leaves unused — three of every
four tokens carry 1–3 such sentinels. The graph passes the buffer to
`ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor`, which forwards
`selected_rows_valid = true`, so every attention kernel level skips its
`row < cache_cap` check and stages the cache at row 0xffffffff. Measured on real
captures (layer 19, chunk 36,864; `DS4_GLM_TRACE_SELECTED=1` reported 319/319
dispatches of a 62k prefill affected): the out-of-range read returned zeros in that
run, so each sentinel acts as a logit-0, value-0 key; against an FP64 masked reference,
zero-sentinel tokens agree at 6.0e-7 normalized RMS while 1/2/3-sentinel tokens differ
by 8.3e-5 / 2.0e-4 / 2.6e-4 (max abs 6.3e-3). Upstream main has the same expander,
wrapper and call, so every legacy reference output contains this behaviour.

Correction on this branch: a checked ragged tail on the blocked-softmax kernel
(`DS4_GLM_DSA_TAIL_CHECKED`, default on), registered in `glm53_exact_mode()` so
`DS4_GLM_EXACT=1` keeps upstream's legacy behaviour by explicit contract. Its quality
assessment is not the 1k manifest (prompts of 240–1,412 tokens at ctx 4,096 never leave
the dense window, so they never execute the corrected kernel) but the 78-case
long-context manifest `bench/fidelity/manifest-long/` (6k/12k/30k/62k-token prompts,
24/24/16/14 cases, ~230-token scored continuations; rebuilt by its `generate.py`, the
cases themselves are not tracked), reported per depth stratum.

## Two Tier 1 items and one opt-in Tier 2 item, recorded here because they are fidelity claims

- **Selector radix fast path** (`DS4_GLM_DISABLE_TOPK_FAST=1` is its only switch):
  bit-identical output, no exact-mode registration; `DS4_GLM_EXACT=1` does *not* clamp
  it. Certified over 150k draws including 16 planted control classes against a CPU
  oracle, 1000/1000 determinism.
- **HC-expand epilogue `ptail`** (`kernel_glm_t2s_q8_hc_expand4_q8_0_ptail`,
  `metal/t2screen.metal`): Tier 1 — the same expression over the same values in the
  same order; only which lane of simdgroup 0 evaluates each `(row, dst_hc)` changes.
  Default on; kill switch `DS4_GLM_DISABLE_HCX_PTAIL=1`. It is not a registry entry, but
  the dispatch policy also suppresses it under `DS4_GLM_EXACT=1`, so **an exact-mode run
  exercises the production HC epilogue and cannot certify ptail**. Its Tier 1 receipt:
  50,000 poisoned draws at each of the three production widths (8192 / 16384 / 12288,
  out_dim 4096) — 3,072,000,000 output words compared word for word, 0 mismatching
  words, 0 poison survivors, 0 non-finite words; 1,000 × 3 determinism repeats, 0
  mismatching words; analytic and planted controls detected on every draw; plus a
  per-dispatch capture of all 48 HC calls of one evaluation, byte-identical.
- **DSA indexer scorer `xr8`** (transposed butterfly reduction across heads): Tier 2,
  **ships opt-in** (`DS4_GLM_ENABLE_SCORER_XREDUCE=1`; depth gate
  `DS4_GLM_SCORER_XREDUCE_MIN_ROWS`, default 37,500 rows; `DS4_GLM_EXACT=1` clamps it
  off). It is not on by default because one 1,024-token block at 300k context produced
  different output bytes from otherwise identical blocks of the same configuration and
  the cause is unknown. Ruled out with their own instruments (1,000 repeats each): the
  kernel on fixed operands, the selector on captured and on xr8's own score rows, the
  whole pipeline replayed from the same prefix, the sampler's tie rule, and eval-to-
  sample ordering. A bounded but unexplained divergence is a fidelity gap; "seen once in
  ten blocks" is not a resolution. A 12-case forced-coverage screen on the long manifest
  (a diagnostic, not certification) measured mean Δ avg_nll +0.0016 (SE 0.0012, 4
  improved / 8 worsened) — inconclusive, and no evidence of equivalence. What would
  change this: the divergence localized and explained (token ids are now retained on
  every benchmark block), and quality evidence at the depths the gate actually ships.

## The pin and the drift ledger

### Custom-weight epoch (historical)

Every row in this section was scored against a custom Q8-KDA requant of the published
Q4_K file that is no longer deployed and is not redistributed; the rows are kept for
provenance only and **cannot be reproduced on the public artifact**. Pinned upstream
commit for this epoch: `ab06d19`. Measured on the Mac Studio M3 Ultra: **avg_nll
0.299642280, first_match 90/100, avg_lcp 9.930** (`bench/fidelity/upstream-ab06d19-epoch.tsv`,
byte-identical to an earlier reference, i.e. that reference was clean upstream).

| build (custom-weight epoch) | avg_nll | Δ vs upstream pin | budget used (of 3e-4) | first_match / avg_lcp |
|---|---|---|---|---|
| upstream ab06d19 (pin) | 0.299642280 | 0 | 0% | 90 / 9.930 |
| fork 5703963, routed-down split OFF, hc_pre wide OFF | 0.299725199 | +8.29e-5 | 27.6% | 90 / 9.930 |
| fork 5703963, routed-down split ON, hc_pre wide OFF | 0.299680649 | +3.84e-5 | 12.8% | 90 / 9.930 |
| fork 5703963 as deployed (split ON, hc_pre wide ON = 16 split-K slices at nsg 32) | 0.299598573 | −4.37e-5 | 14.6% | 90 / 9.930 |
| fork d2b9f79 as deployed (+ router-shared fold [bit-exact], + hc_pre algebra half A at 16 slices) | 0.299643298 | +1.02e-6 | 0.3% | 90 / 9.930 |

Per-change statistics on the 5703963 binary (paired over the 100 per-prompt `avg_nll`
values): routed-down split Δ −4.46e-5, SE 3.56e-5, |Δ|/SE 1.10, wins/losses/ties
52/47/1, p 0.69 — passes S1–S5; hc_pre 16 slices at nsg 32 Δ −8.21e-5, SE 8.03e-5,
|Δ|/SE 1.07, 56/43/1, p 0.23 — passes; hc_pre algebra half A (16 slices) Δ +4.47e-5,
SE 7.94e-5, |Δ|/SE 0.58, 42/57/1, p 0.16 — passes; the same algebra at 32 slices failed
S2 (+1.67e-4): the slice count is itself a Tier 2 knob and the two compound.
`DS4_GLM_EXACT=1` on that build scored 0.299680649, byte-identical to an all-Tier-2-off
build. (An earlier row "split ON = 0.299751615" was a measurement on the wrong worktree
and is withdrawn.)

1k-manifest calibration (gate logs retained outside the public tree): control (all
Tier 2 off) vs BF16 low-rank split-K at 16 tg/tile, 1,000 cases: paired mean Δ −2.807e-4,
SE 3.284e-4, |Δ|/SE 0.85; wins/losses 500/500, p 1.00; first_match 2 → 1; 100-prompt set
0.299643298 → 0.299642818, cumulative vs the pin +5.4e-7. Criterion 2 PASS.

Bundle round 1 — adopted (gate logs likewise retained outside the tree): BF16 low-rank
split-K (16 tg/tile) + DSA blocked softmax (24-row gather, 4-row sub-block, one pass).
Criterion 2 on the 1k manifest vs the reused calibration control: paired mean Δ −2.807e-4,
SE 3.284e-4 PASS; 500/500, p 1.000 PASS; first_match 2 → 1 PASS; 100-prompt cumulative
+5.4e-7 PASS. Speed (interleaved, 4 reps, one binary): 30k 505.35 → 512.49 t/s (+1.4%,
4/4, bytes identical); 62k 495.06 → 502.38 t/s (+1.5%, 4/4, bytes differ as expected for
an FP-order change). Both members are default-on from that round.

The DSA checked-tail assessment on the long manifest also belongs to this epoch; its
logs are retained outside the public tree with the other gate logs.

### Public-artifact epoch

Weights: the upstream-recipe Q4 conversion described in `docs/GLM53_M3ULTRA.md`
(sha256 `828f413cae8ceee74796606814295e00e04c1c7b151aca9a3c5c75db32cb73e0`,
185,299,232,064 bytes). Pinned upstream commit: `9ab7053` (the base of this branch).
Every row is to be filled from a scorer run on that file; nothing from the custom-weight
epoch carries over.

| build (public-artifact epoch) | avg_nll | Δ vs upstream pin | budget used (of 3e-4) | first_match / avg_lcp | receipt |
|---|---|---|---|---|---|
| upstream 9ab7053 (pin), upstream scorer | — | 0 | 0% | — | `bench/fidelity/upstream-9ab7053-epoch.tsv` (to be added from the E1 sanity run) |
| this branch, defaults (fast mode) | — | — | — | — | `bench/fidelity/public-<head>-default.tsv` (to be added from the E1 sanity run) |
| this branch, `DS4_GLM_EXACT=1` | — | measured comparison pending | n/a | — | `bench/fidelity/public-<head>-exact.tsv` (to be added) |

The exact-mode row isolates registered Tier 2 changes; the default row is the fast-mode
drift. The current E6 token-weighted NLL screen did not meet its provisional margin:
candidate 1.270792 versus upstream 1.268859, delta +0.001933 against a +0.0005
criterion. The focused task screen remained 77/77 in both arms (22/23 outputs
byte-identical, one wording difference). Keep both facts; neither substitutes for a
broader equivalence claim.
