# Exact mode: isolate registered floating-point-order changes

## 0. The goal

Exact mode isolates the branch's registered floating-point-order changes from its
other correctness and integration changes. It makes the upstream pin, the same-build
all-off arm and fast mode comparable and attributable. It does not by itself promise
byte identity with upstream: corrected behavior outside the registry can remain active,
and every upstream comparison is reported as measured data.

## 1. The rule

1. Every change that is not bit-exact (Tier 2 in `bench/FIDELITY.md`) ships behind its
   own kill switch (`DS4_GLM_DISABLE_<FEATURE>`) **and** is registered in the exact-mode
   list so the umbrella switch turns it off.
2. Umbrella: `DS4_GLM_EXACT=1` forces every registered Tier 2 feature off. One helper per
   translation unit (`glm53_exact_mode()` in `ds4_metal.m`, `glm53_exact_mode_c()` in
   `ds4.c`) is consulted by each Tier 2 gate; the list of features lives in one place
   with a comment per entry naming the change and the measured Δ.
3. Exact-mode diagnostic: the recipe in section 2, run before deployment and after an
   upstream rebase. Report the exact-mode delta, same-build all-off comparison and fast
   mode next to the pin; do not substitute an assumed `cmp` result.
4. Bit-exact (Tier 1) changes need no flag for exactness (they are exact by construction
   and harness-proven) but keep their kill switches for A/B.

## 1a. Registered items

The authoritative list is the comment block that follows the definition of
`glm53_exact_mode()` in `ds4_metal.m` (search for `Exact-mode registry`); this table is
its index. "Default" is the fast-mode default of the shipped build.

| # | item | switches | default | `DS4_GLM_EXACT=1` |
|---|---|---|---|---|
| 1 | hc_pre split-K slice count (16 instead of the decode default 8) | `DS4_GLM_DISABLE_HC_PRE_WIDE`, `DS4_GLM_HC_PRE_SLICES` | ON | pins the default slice count |
| 2 | BF16 low-rank prefill split-K | `DS4_GLM_DISABLE_BF16_LOWRANK_SPLITK`, `DS4_GLM_ENABLE_BF16_LOWRANK_SPLITK=0` | ON | off |
| 3 | Blocked online softmax in the batched sparse DSA attention kernel | `DS4_GLM_DISABLE_DSA_BLOCKED_SOFTMAX` | ON (stage 24, sub-block 4, one-pass) | off |
| 4 | Bounds-checked ragged tail (`TAILCHK`) | `DS4_GLM_DSA_TAIL_CHECKED=0`, `DS4_GLM_DISABLE_DSA_TAIL_CHECKED=1` | ON | off — exact mode keeps the *legacy* unchecked tail by explicit contract, because exact mode reproduces upstream main and upstream main has the same unchecked tail |
| 5 | DSA indexer scorer, transposed butterfly across heads (`xr*`, shipped shape `xr8`), with the depth gate | on `DS4_GLM_ENABLE_SCORER_XREDUCE=1`; shape `DS4_GLM_SCORER_XREDUCE_SHAPE`; gate `DS4_GLM_SCORER_XREDUCE_MIN_ROWS` (default 37500, 0 disables); kill `DS4_GLM_DISABLE_SCORER_XREDUCE=1` | **OFF (opt-in)** | off unconditionally (the clamp is in `ds4_gpu_glm_scorer_variant_index()`, so even an explicit `DS4_GLM_SCORER_VARIANT=xr*` cannot escape it) |
| 6 | hc_pre algebra halves A / B | `DS4_GLM_DISABLE_HC_PRE_ALGEBRA_A`, `..._B` | A ON, B OFF | off |
| 7 | split_group8 softmax block rows | `DS4_GLM_SPLIT8_BLOCK_ROWS_SHALLOW`, `..._DEEP` | 32 / 128 | pins 32 / 128 |
| 8 | Q8_0 matvec simdgroup count (`Q8NSG`), globally and per family | `DS4_METAL_Q8_MV_NSG` (upstream knob), `DS4_GLM_T2S_Q8NSG_<FAM>` | shipped default (4, or 2 under TP2) | pins the shipped default |
| 9 | BF16 matvec simdgroup count | `DS4_GLM_T2S_BF16_NSG` | shipped default | pins the shipped default |
| 10 | split8 double-buffered staging | `DS4_GLM_ENABLE_SPLIT8_DBLBUF=1` | OFF (opt-in) | off |
| 11 | split8 reduce v-plane simd_sum | `DS4_GLM_ENABLE_SPLIT8_VPLANE=1` | OFF (opt-in) | off |

Entry 5 in more detail: `xr8` computes the same real-valued score as the production
scorer (same heads, same F32 query, same F16 pooled keys, same head weights, same
`max(dot*scale,0)*w` formula) and changes only the summation order — the 32 per-head
`simd_sum` trees become one transposed butterfly over the simdgroup, and the 32-step
sequential head accumulation becomes a 32-leaf tree. It is registered even though it is
off by default so that turning it on cannot silently escape the umbrella. The depth gate
keeps the production scorer on calls with fewer rows than the threshold, where `xr8`
measured slower; it is evaluated per call and only ever substitutes the bit-exact
production kernel for a Tier 2 one, never the reverse. The default threshold is a
documented convention between the measured loss (62k context, 15,542 rows) and the
measured win (300k, 74,998 rows), not a measured crossover. Why it stays opt-in is in
`bench/FIDELITY.md`.

### Two Tier 1 items that are not registry entries, and what exact mode does with them

- **Selector radix fast path** (`ds4_gpu_glm_topk_fast_enabled()`, `metal/argsort.metal`),
  default on, kill switch `DS4_GLM_DISABLE_TOPK_FAST=1`. Bit-identical, therefore no
  exact-mode registration; **`DS4_GLM_EXACT=1` does not clamp it.**
- **HC-expand epilogue `ptail`** (`kernel_glm_t2s_q8_hc_expand4_q8_0_ptail`,
  `metal/t2screen.metal`), default on, kill switch `DS4_GLM_DISABLE_HCX_PTAIL=1`.
  Tier 1, not a registry entry, but the dispatch policy additionally suppresses it under
  exact mode (`ds4_metal.m`, the `DS4_GLM_DISABLE_HCX_PTAIL` site: the production
  epilogue is selected when either the kill switch or `glm53_exact_mode()` is set). An
  exact-mode run therefore exercises the production HC epilogue and cannot certify
  ptail; ptail's evidence is the fast/default arm with its own dispatch proof and the
  randomized Tier 1 receipt summarized in `bench/FIDELITY.md`.

## 2. The exact-mode check

There is no `bench/exact-check.sh`; the check is the following recipe, run by hand, and it
reports **three separate results**, never merged into one verdict. `S` is the scorer
built from this tree (`make -C gguf-tools quality-score`), `M` the model file, `MAN` the
100-prompt manifest `gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv`,
and `PIN` the TSV produced by the same scorer source built from clean upstream at the
pinned commit on the same `M` (kept in `bench/fidelity/upstream-<pin>-epoch.tsv`).

```sh
S=gguf-tools/quality-testing/score_official
M=gguf/GLM-5.3-Flash-Q4_K.gguf
MAN=gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv
export DS4_GLM53_MEMORY_CEILING_GB=400

# 1. Upstream comparison -- a measured diagnostic.
DS4_GLM_EXACT=1 $S $M $MAN exact.tsv
python3 gguf-tools/quality-testing/compare_scores.py \
  bench/fidelity/upstream-<pin>-epoch.tsv exact.tsv

# 2. Same-build individual-kill-switch identity -- the umbrella and the
#    individual switches agree.  Every registered kill switch, plus
#    DS4_GLM_DISABLE_HCX_PTAIL=1 (which exact mode also applies), and
#    WITHOUT DS4_GLM_DISABLE_TOPK_FAST (which it does not).
DS4_GLM_DISABLE_HC_PRE_WIDE=1 DS4_GLM_DISABLE_BF16_LOWRANK_SPLITK=1 \
DS4_GLM_DISABLE_DSA_BLOCKED_SOFTMAX=1 DS4_GLM_DISABLE_DSA_TAIL_CHECKED=1 \
DS4_GLM_DISABLE_SCORER_XREDUCE=1 DS4_GLM_DISABLE_HC_PRE_ALGEBRA_A=1 \
DS4_GLM_DISABLE_HC_PRE_ALGEBRA_B=1 DS4_GLM_DISABLE_HCX_PTAIL=1 \
  $S $M $MAN alloff.tsv
cmp exact.tsv alloff.tsv                                    # must be silent

# 3. Fast mode, reported next to it (drift vs the pin, budget used).
$S $M $MAN fast.tsv
python3 gguf-tools/quality-testing/compare_scores.py bench/fidelity/upstream-<pin>-epoch.tsv fast.tsv
```

A third comparison, **prior-deployment identity**, is only valid against a retained
exact-mode or all-off TSV of the intended build lineage; a fast-mode TSV with Tier 2
features enabled is not a valid reference for it, and when no qualifying artifact
exists the comparison is *unavailable*, not a pass.

The 3e-4 threshold of `bench/FIDELITY.md` is the fast-mode cumulative-drift budget.
If the upstream comparison differs, classify it as a change outside the registry or a
new registered-change failure. Checks 2 and 3 distinguish those cases. Record the
measured difference and its quality effect rather than treating any nonzero byte delta
as a generic failure.

## 3. Prefill

The same rule applies to prefill-side changes: every Tier 2 prefill change is registered
in the exact-mode list, and the check above covers prefill automatically because the
scorer's own prefill runs through the changed kernels.

## 4. Upstream drift

The upstream comparison is against the *pinned* commit. When the branch rebases onto a
newer upstream, re-score clean upstream at the new commit, re-run the diagnostic, and update
the pin in `bench/FIDELITY.md` by an explicit decision. Upstream's own numerics changes
(they merge FP-order changes too) are then part of the new baseline. The current pin is
`9ab7053`, the base of this branch; the custom-weight-epoch ledger in `bench/FIDELITY.md`
was pinned to `ab06d19` and is historical.
