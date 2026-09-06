# Benchmarks and fidelity harness for the GLM-5.3-Flash Metal branch

This directory holds the instruments used to gate every change on this branch, the
fixtures they score against, and the ledgers that bind the branch's numerics and speed
to upstream. What the branch changes and why is in `CHANGES-GLM53.md` at the repository
root; how to build, run and reproduce the headline configuration is in
`docs/GLM53_M3ULTRA.md`; the fidelity rules are in `FIDELITY.md` and the exact-mode
diagnostic in `EXACT-MODE-PLAN.md`; the current release ledger is in `RELEASE-EVIDENCE.md`.

## Why the operating points are deep

The benchmark contexts were chosen from where real decoded tokens happen. A six-day
coding-agent thread on this machine (76 turns, 11+ compactions, 5,226 requests, 3.65M
decoded tokens in the serving logs) placed **97.4% of decoded tokens at ≥ 40k context**
(0–10k: 1.2%, 10–40k: 1.4%, 40–80k: 5.0%, 80–120k: 18.2%, 120–160k: 17.2%, 160k+: 57.0%).
97.6% of requests were incremental (median 111-token tool-result prefill on a hot KV
prefix, generating ~700 tokens); 2.4% were ~47k compaction re-prefills. Benchmarks at
2–8k context therefore measure a regime that barely exists in agentic use, and the
instruments below are built around 50k–300k.

## Instruments

| script | what it measures | run |
|---|---|---|
| `quickbench.py` | serving-path decode and prefill t/s at 50k and 100k agent-shaped context: 1 cold prefill + 5 incremental turns through `ds4-server`, KV cache and incremental prefill behaving exactly as in serving; requests that miss the KV prefix are flagged, never averaged in | `python3 bench/quickbench.py [configs...]` — env in the docstring (`QUICKBENCH_*`) |
| `prefill-gate.sh` | interleaved A/B prefill speed gate: base, variant, base, variant … per prompt bucket (62k, 30k, 512, 128, 111 tokens), prefill t/s, seconds to first token, and byte identity of the 16-token greedy output between arms | `A_BIN=… B_BIN=… B_ENV="DS4_…=1" REPS=4 BUCKETS=62k,30k bench/prefill-gate.sh` |
| `tier2-gate.sh` | prefill Tier 2 criterion 2: scores control and candidate (same binary, candidate env on) on the 1k manifest and the 100-prompt pin set, then compares | `CAND_ENV="DS4_…=1" bench/tier2-gate.sh` |
| `compare_1k.py` (+ `test_compare_1k.py`) | the verdict for the above: paired mean Δ, SE, |Δ|/SE, exact binomial on wins/losses, first_match, cumulative drift on the 100-prompt set | called by `tier2-gate.sh`; `python3 bench/test_compare_1k.py` runs its unit tests |
| `fidelity/manifest-1k/generate.py` | rebuilds the 890-case equivalence manifest (public-domain passages and upstream's long-context test prompts; provenance per case) | see its docstring for the untracked corpora |
| `fidelity/manifest-long/generate.py` | rebuilds the 78-case long-context manifest (6k/12k/30k/62k-token prompts) used to assess the DSA checked tail; the cases are not tracked, so run it before scoring | same corpora, plus a built `ds4` and the model file for its tokenizer (no GPU) |

All scripts take the model path, results directory, prompt directory and any GPU-lock
helper from the environment (defaults are relative to the repository; see each
script's header). The prompt buckets for `prefill-gate.sh` are plain-text files of the
nominal token count and are not tracked; any long natural-language text works, and the
62k bucket is the "needle" prompt used in the native reference recipe below.

Every gate assumes exclusive use of the GPU: no `ds4-server`, no other model process,
no concurrent Metal micro-benchmarks. Single-shot gates taken under contention produced
regressions of several t/s that vanished when re-run interleaved in a quiet window, so
the protocol is ≥ 4 interleaved repetitions and a number is discarded if any other model
process appeared during it.

### The scorer

Quality is measured with upstream's teacher-forced scorer:

```sh
make -C gguf-tools quality-score          # rebuild after EVERY source change
DS4_GLM53_MEMORY_CEILING_GB=400 gguf-tools/quality-testing/score_official \
    gguf/GLM-5.3-Flash-Q4_K.gguf \
    gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv out.tsv
python3 gguf-tools/quality-testing/compare_scores.py control.tsv out.tsv
```

The 100-prompt set is the upstream-pin drift ledger; the 1k manifest under
`fidelity/manifest-1k/` is the Tier 2 quality gate; the long manifest under
`fidelity/manifest-long/` is the depth assessment. `fidelity/*.tsv` are retained scorer
outputs (see the ledger in `FIDELITY.md` for which epoch each belongs to).

### The native reference recipe (throughput)

The quoted native records use the ordinary single-session CLI generation loop, cold
prefill and greedy decoding. Verify the public model and prompt digests against
`RELEASE-EVIDENCE.md`; long prompt files are release inputs and are not silently
substituted:

```sh
export MODEL=gguf/GLM-5.3-Flash-Q4_K.gguf
export PROMPT62=/path/to/needle-64k.txt
export PROMPT300=/path/to/prompt-300k.txt
export DS4_GLM_GEN_COUNTERS=1 DS4_GLM_IGNORE_EOS=1

./ds4 -m "$MODEL" --metal --nothink --temp 0 \
      -c 70000 -n 2048 --prompt-file "$PROMPT62"
./ds4 -m "$MODEL" --metal --nothink --temp 0 \
      -c 320000 -n 2048 --prompt-file "$PROMPT300"
```

`DS4_GLM53_MEMORY_CEILING_GB=280` remains an optional conservative admission cap for
the 512 GB machine, but it was absent from the final default receipts. The prefill chunk
already defaults to 8192. The final guarded-profile build resolves its model mapping and
expert-bank schedule without force environment variables; the `b723dfa` receipts used
explicit bank, fused, pipeline, `xr8` and untracked-view controls and are labeled
accordingly.

To reproduce the named `b723dfa` capability configuration before running either command
above, add:

```sh
export DS4_METAL_MODEL_UNTRACKED=1 DS4_GLM_ENABLE_EXPERT_BANK=1
export DS4_GLM_EXPERT_BANK_FUSED_LAYER_CB=1
export DS4_GLM_EXPERT_BANK_PIPELINED_LAYERS=1
export DS4_GLM_ENABLE_SCORER_XREDUCE=1
export DS4_GLM_DISABLE_ROUTER_SPLITK_B4=1
```

A block is valid only when the native generation record reports `n_generated=2048`,
`n_decode_eval=2047`, and `stop=predict_limit`. The session record also reports
`generated=2048`, `requested=2048` and 2,048 committed forward positions. One complete
block records achieved capability. Interleaved matched runs are required to assign a
speed difference or quantify repeatability.
`DS4_GLM_IGNORE_EOS=1` is a benchmark-only fixed-horizon control and changes the text.
With no `--dflash` weights, bare startup resolves serial mode and allocates no drafter.

## Results

### Public-artifact release epoch

The final `538c37c` artifact and the earlier `b723dfa` capability build use the public
185,299,232,064-byte GGUF at sha256 `828f413c…`. The complete identity, scope and open
rows are in `RELEASE-EVIDENCE.md`; compact final receipts are under
`receipts/glm53-m3ultra/`.

| final measurement | result | scope / receipt |
|---|---:|---|
| 62,174-token native prefill | **550.27 t/s** | bare guarded bank/fused/pipeline8/untracked policy; `final-native62.json` |
| same-prompt native serial decode | **37.868944385 t/s** | `n_generated=2048`, `n_decode_eval=2047`, `stop=predict_limit`; same receipt |
| 33,148-token bank boundary | **556.73 AUTO / 540.64 OFF t/s** | same 77 output bytes; `final-bank-boundary33.json` |
| affected server runtime | **16/16 plus 4/4 pipeline/SIGTERM passed** | `final-runtime.json` |
| routed failure/re-prime | **comparator v3 passed** | 128 native tokens in both arms, identical 183-byte response, zero cache |
| 300,000-token native prefill | **473.64 t/s** | five bank groups, 210 routed layers, no refusal; `final-native300.json` |
| 300,000-token native serial decode | **37.187023218 t/s** | complete 2,048-token native block; same receipt |

Earlier `b723dfa` capability rows:

| measurement | `b723dfa` capability | scope / receipt |
|---|---:|---|
| 62,174-token native prefill | **550.72 t/s** | forced bank + fused layer command buffer + pipeline; `ASTRAL-BANK-PIPELINED62-20260906T194550Z` |
| 300,000-token native serial decode | **37.2993996 t/s** | 2,048 generated / 2,047 evaluated, limit stop; `BASE300-astral-b723-pipeline-20260906T195448Z` |
| 300,000-token native prefill | **473.75 t/s** | same receipt; five bank groups admitted |
| SQL 8,192-token fixed horizon, serial / conservative / speculative | **38.5850 / 47.2064 / 60.7875 t/s** | favorable repetitive fixture, all arms 22,036 bytes, truncated during tuple 483; `ASTRAL-THREE-MODES-20260906T192241Z` |
| focused task screen | **77/77 in both arms** | 22/23 byte-identical, one wording difference; `TASKCHECK-20260906T154355Z` |
| E6 token-weighted NLL criterion | **unmet** | delta +0.001933 against upstream; provisional +0.0005 criterion not met |

The 300k generated block matched the retained v3+`xr8` reference block. This is
lineage evidence rather than a universal upstream-identity claim. Exact mode disables
registered Tier 2 changes but leaves the corrected DSA pad-row semantics active; compare
its measured scores instead of requiring an upstream TSV `cmp` pass.

### Custom-weight epoch (historical)

Everything below was measured on builds that predate the merge onto upstream `9ab7053`,
on a custom Q8-KDA requant of the published Q4_K file that is no longer deployed and is
not redistributed, and on this machine's serving stack. The tables are kept because they
document *what each change bought when it landed*; they are not reproducible on the
public artifact and are not the branch's claim. Decode t/s is the serving path
(`quickbench.py`) unless marked "CLI".

**Serving path at depth** (50k / 100k agent-shaped contexts, thinking on, seed 12345):

| build | 50k decode t/s | 100k decode t/s | prefill t/s | incremental KV cache |
|---|---|---|---|---|
| pre-campaign upstream-era build, published Q4_K | 19.9 | 19.5 | 381 / 372 | hits at 50k; **all 5 turns miss at 100k** |
| after the first decode rounds (dispatch widening, epilogue fusions) | 33.1 | 32.8 | 371–385 | hits |
| after the KDA pack / router fold / hc_pre rounds | 36.8 | 36.5 | 378 / 373 | hits |
| upstream head of that week, same custom file | 23.1 | — (every turn re-prefilled) | 384 / 372 | misses |

The 100k cache misses on the older builds were a server bug, not a kernel property:
after a thinking turn the server remembered a "visible key" rendered in DeepSeek syntax,
which GLM's template never matched, so every multi-turn thinking exchange with a
reasoning-stripping client re-prefilled from scratch. That fix, and the KV checkpoint
policy that survives mid-transcript mutations, are described in `CHANGES-GLM53.md`
("Server").

**CLI single-stream decode** on the same custom file, short context (GSM prompt, 512
tokens, greedy): 21.6 t/s at the start of the campaign, 29.2 for upstream of that week,
40.1 at the end of the decode rounds (all outputs byte-identical to upstream's in the
Tier 1 rounds). At depth on the CLI: 62k 33.9 → 37.7 t/s, 150k 36.8 → 37.3, 300k 35.3 →
36.7 (the last two from the DSA top-k radix fast path alone).

**Prefill** (62k bucket, interleaved 4-rep gate): 384.6 t/s at the start of the prefill
work, 502.4 t/s after bundle round 1 (BF16 low-rank split-K + blocked DSA softmax),
with the 30k bucket 505 → 512 t/s and byte-identical output.

**Speculative decoding on the serving path** (same epoch): DFlash2 tracked plain serial
decode at depth (33.0 vs 33.1 t/s at 50k) and won +25–45% on math/structured/JSON
generation; the embedded MTP-2 draft (`--mtp`) lost 10–20% at depth on agentic content
because the multi-token verify pass does not use the single-token decode graph's
fusions. See `docs/GLM53_M3ULTRA.md` for DFlash2's status on the merged branch.

## Campaign notes (custom-weight epoch, summarized)

The working notes of the decode and prefill campaigns are not part of the public tree
(they were written as dated journals against machine-local paths). What follows is
their technical content; every number is from the custom-weight epoch above and is
kept for orientation, not as a claim about the public artifact.

### Prefill

Prefill is compute-bound, not bandwidth-bound: at 62k tokens the shipped build moves
the equivalent of several TB/s of decode-style weight traffic, so the wall is the
chip's matrix-multiply throughput on dequantized operands and none of the decode
levers (dispatch folds, matvec load widths, hc_pre widening) transfer. Two regimes
matter: long cold prefill (thousands of tokens; GEMM-bound; the compaction-and-reload
event a coding agent's user waits through) and short incremental prefill (~100-300
tokens appended to a cached prefix; latency-bound, ~3× below the long regime per token
because the batch never fills the matrix units).

The mode-2 kernel ledger at 62k (fast mode, 1,964 µs/token = 509 t/s at the time)
attributed the cost by family, with the excess measured against a matrix-unit anchor
of 1,329 µs/token: dense Q8 stack 577 µs (excess 46), routed gate+up 518 (101), DSA
sparse attention 266 (163), routed down 253 (44), indexer scorer 67 (42), KDA prefill
quartet 55 (47). Levers shipped from that map, each Tier 1 unless noted and each behind
the kill switch named in `docs/GLM53_M3ULTRA.md`: BF16 low-rank split-K and the blocked
DSA softmax (Tier 2, "bundle round 1"), the wide 16-byte gather in the blocked kernel,
the checked ragged tail (Tier 2, registry entry 4), the indexer head-fold scorer and
causal score grid, the dense per-chunk half copy and its three-slot ring for the KDA
q/k/v group, the small-kernel folds (width-4 HC expand, folded FFN residual add,
parallel routed work map, width-4 shared SwiGLU), and the register-resident HC row
chain. The 62k prefill went from 384.6 t/s to 529.6 (fast) / 514.3 (exact) across the
campaign; the 30k bucket tracked it.

The campaign closed on evidence that 600 t/s at 62k is not reachable within the
fidelity rules on this chip and model: the 636 µs/token above the anchor split into
attention ~160 (gather-bound; adjacent tokens share only 58% of selected rows on
average, so token-shared attention cannot pay), tail ~190 (about 57 recoverable) and
GEMM taxes ~280 (about 100 recoverable). Structural routes measured and closed:
token-shared attention, unfused and three-pass attention, prefetch into the SLC,
resident weight banks, and row-major or tile-major expanded expert layouts (+53-60
µs/token, under the 80 µs bar that a layer-major graph rewrite would have to clear;
combined with whole-prompt routed batching it crosses the bar at ~+85 µs/token, i.e.
~554 t/s conditional, at the price of a layer-major graph, per-layer KDA/conv
snapshots and ~92 GiB of activation scratch at 300k — recorded, not built).

### Decode

At the 8k gate the serial decode ran at 24.8 ms/token against a 20.17 ms byte floor
(14.22 GB/token at 705 GB/s STREAM), i.e. ~81% of the bandwidth wall. The ledger put
the remaining ~4.6 ms/token in: the hc_pre latency chain (90 dispatches, ~1.6 ms),
the four large MoE dispatches' ramp and tail (~1.5 ms; overlapping them did not pay),
the latency-bound small DSA kernels (~1.6 ms; at depth the split8 partial, the argsort
top-k chain and the value projection dominate), the KDA glue prologue (0.34 ms) and
command-buffer idle (0.46 ms; the flush-interval sweep was flat). Remaining levers,
ranked: the split8 partial's block rows at depth (Tier 2), the top-k finisher (two
8-bit digit passes), the hc_pre tail reduce prologue, routed gate+up lane reassignment
(Tier 2), more KDA glue rows in flight, a coalesced Q8_0 value projection inside the
split8 reduce (Tier 2), and a float4 `indexer_pool_update` prologue — together
~+0.7-1.0 t/s at 8k and ~+1.2-1.6 at 62k if all land; beyond that the residue is the
dependent-latency structure of the model (mHC, DSA). Measured dead, do not retry:
persistent or dataflow kernels of any kind, SLC prefetch (~33 MB SLC, no overlap in
the serial encoder), encoder reorders for overlap, recomputing the router selection
per expert threadgroup, hc one-dispatch or ticket forms, merging small independent
dispatches for its own sake, requantizing the F32 router weights, and fewer Sinkhorn
iterations (a model change, not a floating-point-order change).

Decode at depth on the CLI: 62k 37.6 t/s (exact 37.2), 150k 36.8, 300k 35.3 — a 6.2%
decline, +1.76 ms/step. Decode-only ledgers attributed it to the top-k merge chain
(+1.19 ms/step) and the DSA indexer scorer (+0.75) with everything else flat, which is
why the bounded-radix top-k fast path was built (+1.29 t/s at 300k, +0.47 at 150k,
neutral at 62k). A compaction summary at ~290k context that looked like a decode
slowdown in the serving log was storage: a 7 GB KV checkpoint write plus two disk
evictions inside the generation window, with the per-chunk decode rate flat.
The DFlash2 drafter's batched sparse-FFN route does not reach the router-shared fold
and the routed-down split; porting them there is the one speculative-decode lever
identified. The live-KV checkpoint after truncated and tool-context turns is in
`CHANGES-GLM53.md` ("Server").

### Exact-reference gate (E1 / E2)

`FIDELITY.md`'s Tier 2 rule for prefill-side changes needs an exact answer rather than
a scorer estimate: on the 100-prompt manifest a prefill floating-point-order change has
a paired standard error of 5.5-6.3e-4, above the scorer's ceilings, so the scorer can
prove neither harm nor safety. There is one model and one chip, so the exact error is
computed instead — per kernel (E1) and end to end (E2). Nothing here relaxes the
absolutes: every non-bit-exact change still ships behind its kill switch and is
registered in `glm53_exact_mode()`, and the exact-mode check of `EXACT-MODE-PLAN.md`
§2 must still reproduce upstream's TSV byte for byte.

**E1.** A driver (not part of this repository) builds the same Metal library the
production binary builds — the `ds4_metal.m` prologue plus `metal/*.metal` in
`required_sources` order — so every arm is the real kernel, and computes two FP64
sequential references per draw: REF_STAGE applies the kernel's own staging semantics
exactly (for the BF16 family, BF16 → half for the weight and float → half for the
activation, which is what the tile's B stage does) and then sums in double, isolating
the accumulation order, the only thing a reassociation changes — the verdict is taken
on this column; REF_EXACT narrows nothing and is reported so the staging floor
(`|REF_EXACT − REF_STAGE|`, a property of the kernel family) is visible. Pass: the
candidate's max and mean |Δ| ≤ production's on REF_STAGE at every production shape.
The driver refuses vacuous proofs, which two earlier levers had shipped: the finite
fraction must be ≥ 20% (a NaN/Inf poison over a 4096- or 16384-long reduction makes
the comparison a tautology), a known-different control arm (the same split-K kernel at
the maximum slice count) must actually differ, and every destination is poisoned with
its own NaN payload per arm so an unwritten word shows as a difference. Two fills, both
required: a wide six-decade amplitude mixture with one planted NaN/Inf in 20% of
activation columns and every 24th weight row, and a narrow uniform fill closest to a
normalized activation. Adding a kernel pair means a case table of production shapes,
three arms encoded exactly as the host encodes them (same function constants,
threadgroup memory and grid), and a reference written from the kernel's semantics, not
its code; a softmax reference must reproduce the kernel's max-subtraction convention or
it measures the convention instead of the order.

**E2.** The reference is the production GPU forward pass with the touched kernels
replaced in place by an FP64 host computation, because a CPU forward pass is not
available for GLM-5.3: `--metal-graph-prompt-test` is the DeepSeek raw/SWA harness and
compares the final position only, `--first-token-test` has no KDA branch and
dereferences NULL on layer 0, `matvec_any` has no BF16 dot product, the KDA recurrence,
hyper-connection mixer and pooled DSA indexer exist only on the GPU side, and the
engine refuses a non-graph backend for the family. On this branch the in-tree host
reference of that kind is `DS4_GLM_DSA_ATTN_FP64_REF` (DSA attention; minutes per
prompt); the BF16 low-rank replay used for bundle round 1 is not carried in the public
tree. A host replay synchronizes the GPU to read activations back, which ends the
concurrent command batch, so every E2 arm runs with `DS4_KDA_CONCURRENT_DISABLE=1` (a
scheduling choice only, measured byte-identical on and off). Logits come from the CLI's
`--dump-logits FILE` (all next-token logits at `%.9g` plus the argmax, written before
generation); positions are sampled by truncating the prompt with
`DS4_GLM_PREFILL_TRUNC=N`, one run per position, which is also how the dense window
(truncations at 1024 and 2048) and the sparse DSA regime (4096 and above) are both
covered; DSA selection indices come from `DS4_METAL_GRAPH_DUMP_PREFIX` with
`DS4_METAL_GRAPH_DUMP_LAYER=3` (the DSA layers are `il % 4 == 3`; 2,051 entries per
token row in the sparse regime); per-layer hidden rows from `DS4_GLM_HIDDEN_DUMP` for
bisecting a divergence. The prompt set is four prompts of ≥ 8k tokens, one ≥ 30k, one
of them compaction-shaped (a tool-schema block, a long tool-using conversation, then a
"summarize so far" turn) — the shape that produces most of the prefill tokens a
coding-agent server actually sees. Three arms per configuration (reference,
production, candidate); pass: candidate ≤ production on max and mean |Δ logit| and
candidate's selection-index agreement ≥ production's, both arms always reported
against the reference. A harness should verify its first dump and abort if it is
missing rather than discover a silent no-op after holding the GPU for an hour.

**Not covered by E1/E2.** Graph state (a kernel harness cannot see a shared scratch
buffer overwritten by the next call in a concurrent encoder — the split-K partials
once raced exactly that way while the isolated harness reported 1000/1000
deterministic; keep the cheap repeated-run byte check as a separate early gate); the
scorer's other jobs (symmetry, cumulative drift vs the upstream pin and `first_match`
still gate; re-measure the control on the same binary); and speed (unchanged: the
interleaved ≥ 4-rep gate).

## Fixtures

- `fidelity/manifest-1k/` — 890 prompt/continuation pairs (Shakespeare and *I
  promessi sposi* from Project Gutenberg, upstream's long-context test prompts and a
  deep-math set) with `provenance.json` and `manifest.tsv`. The generator's sixth
  source — 110 cases cut from this repository's Markdown, ids `case_0890`–`case_0999` —
  is retired because the documents it sampled were campaign working notes that are not
  in the public tree; it was generated last, so `generate.py` reproduces the 890 kept
  cases unchanged. Retained scorer outputs from that epoch may still carry rows for the
  retired ids.
- `fidelity/manifest-long/` — `generate.py` and the `provenance.json` of the original
  generation only; the 78 cases are rebuilt with `python3
  bench/fidelity/manifest-long/generate.py` (needs `bench/fidelity/corpus/shakespeare.txt`,
  `speed-bench/promessi_sposi.txt`, a built `ds4` and the model file, whose tokenizer it
  uses on the CPU). The 75 public-domain cases come back as recorded in the provenance;
  the three `repo_docs` cases are cut afresh from the public documentation.
- `fidelity/*.tsv` — retained 100-prompt scorer outputs, named by build and switch state.
- The scorer TSVs and logs of the calibration, bundle-1 and checked-tail gates, and
  the `quickbench.py` result directories of the custom-weight epoch, are kept outside
  the public tree (they carried machine-local paths); the verdict numbers are quoted
  in `FIDELITY.md` and in the historical tables above.
