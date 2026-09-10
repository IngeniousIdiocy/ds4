# Changes on the GLM-5.3 M3 Ultra branch

An engineering account of what this branch changes relative to upstream ds4 at
`9ab7053`, written for a reviewer of the code. User-facing details (configuration,
switches, status, known issues) are in `docs/GLM53_M3ULTRA.md`; the fidelity rules in
`bench/FIDELITY.md`; the exact-mode diagnostic in `bench/EXACT-MODE-PLAN.md`. Current
capability numbers are kept in `bench/RELEASE-EVIDENCE.md`, where the earlier
`b723dfa` capability build is separated from the final `538c37c` artifact. The historical
per-round numbers, measured on a retired custom quantization, are kept only in
`bench/README.md` under a historical heading.

The target is one machine and one regime: single-stream GLM-5.3-Flash on an M3 Ultra,
at the 40k–300k contexts where a coding agent spends 97% of its decoded tokens. The
M3 Ultra is two dies and 80 GPU cores behind roughly 700 GB/s of measured memory
bandwidth; a decode step of this model reads about 14 GB of weights, so the roofline is
~50 t/s and the work was to close the gap to it without changing what the model
computes.

## 1. Decode kernels

Where the time went was established first, with a stage-ablation switch
(`DS4_GLM_DECODE_ABLATE` family) and a per-kernel bandwidth ledger
(`DS4_KERNEL_LEDGER`, `ds4_metal.m:1770`). Two findings drove everything after:

- The large kernels (LM head, KDA qkv, dense FFN) already ran at 700+ GB/s. The debt
  was in **small, parallelism-starved kernels**: hundreds of dispatches per token with
  1–64 threadgroups on an 80-core GPU, each paying a fixed dependent-dispatch floor. Of
  ~1,000 dispatches per token, ~600 under 100 threadgroups consumed a fifth of the token
  for 4% of the bytes.
- A dependent-dispatch boundary in the production serial encoder is worth about 1 µs,
  not the 6 µs the ledger's intercept first suggested, so *launch* fusion alone is not
  the lever; removing a dependency hop, widening a starved kernel, or folding a small
  latency-bound row set into the tail of a large streaming grid is.

What shipped, all default-on, each with a kill switch, and bit-identical unless marked
Tier 2 (see `bench/FIDELITY.md` for what each tier proves):

- **KDA layers** (`metal/glm53_kda.metal`, host in `ds4.c:43661-45473`): the recurrence
  split into phases (64 → 512 threadgroups); prep and state fused into one wide
  threadgroup per head (two dispatches per layer instead of three); the low-rank
  projections packed — `f_a+g_a+beta` as one flat matvec, `f_b+g_b` as a pair — and the
  flat rows folded into the tail of the Q8_0 q/k/v grid so they ride in the same wave
  instead of paying their own launch; `f_b/g_b` and the output projection folded into
  the state kernel's prologue and tail. An all-Q8 fusion of the whole projection stage
  (eight dispatches to two) exists opt-in (`DS4_GLM_ENABLE_KDA_PROJ_FUSE`) and is the
  first path written for the public artifact's all-Q8 KDA layout; it is not validated.
- **Hyper-connection (HC) pre-stage** (`metal/dsv4_hc.metal`, `ds4_metal.m:52600-53110`):
  the RMS + split-K mixer and the reduce + weighted-sum collapsed into a fused pair with
  a redundant-RMS trick (four dispatches to two); the tail replicated across 13
  threadgroups, each writing its own slice, with zero cross-threadgroup data; the
  split-K reduce in float4; **Tier 2:** 16 split-K slices at 32 simdgroups instead of 8,
  and "algebra half A" (the unscaled split-K dot with the scale applied in the tail, so
  no threadgroup re-reads the 64 KB HC row). The one-dispatch form with a
  last-threadgroup tail was built and measured slower (the tail is single-core-bandwidth
  bound, not dispatch bound) and is kept opt-in.
- **Epilogue fusions** (`ds4.c:44979-46471`): residual add and mHC expand folded into
  the epilogue of the preceding Q8_0 matvec (MoE shared-down, attention-out, dense-down),
  removing ~130 dispatches per token and, more importantly, a dependency hop each; HC
  norm and norm+mix fused; the Sinkhorn comb moved past a barrier it did not need.
- **Router and shared expert** (`ds4_metal.m:41929-42160`, `metal/dense.metal`,
  `metal/dsv4_misc.metal`): top-8 selection folded onto the tail of the router logits
  matvec; the router then folded into the head of the shared-expert gate+up grid (144
  router threadgroups ahead of 512 shared-expert ones, with a ticket and seq_cst fences
  on both sides, and a new 8 KB shared-mid tensor that breaks an `ffn_mid` alias) — 42
  fewer dispatches per token and the largest single decode win; an iterative top-k
  router selection instead of a 512-wide bitonic sort.
- **Routed experts** (`metal/moe.metal`, `ds4_metal.m:42893-42940`): **Tier 2**
  expert-parallel down projection — one expert slot per threadgroup (16,384 threadgroups
  where the sequential form launched 1,024), partials summed in fixed slot order inside
  the shared-down epilogue; `ushort4` quant loads in the gate+up pair.
- **DSA attention and indexer** (`ds4.c:42366-44626`, `ds4_metal.m:20112-20540`,
  `metal/argsort.metal`, `metal/dsv4_misc.metal`): q_a, kv_a and the three indexer
  projections in one dispatch with the BF16 rows at the head of the Q8 grid; the
  indexer scorer stages its query once per threadgroup (it had been re-reading 249 MB of
  query per token against 4 MB of keys); the mono → split8 attention crossover lowered
  to 64 selected rows, the split8 reduce widened to 512 threads with ushort-paired Q8_0
  value rows; and, for the top-k that selects the 2,051 attended rows, a **bounded
  radix fast path** (three dispatches with a 13-bit signed-monotone key, a predicate
  over the complete input that accepts only a strictly separated winner set, and the
  production sort/merge chain always encoded as indirect dispatches whose grids the fast
  path writes — zero threadgroups on accept, real grids on reject). The last item is what
  the 300k regime gains from; acceptance is ~99% on real score rows.
- **Q8_0 matvec family:** the residual-add/HC epilogue fusion above; a per-family
  simdgroup-count override (`DS4_GLM_T2S_Q8NSG_<FAM>`, Tier 2, pinned under exact mode).
- **HC-expand epilogue `ptail`** (`metal/t2screen.metal`): the epilogue spread over
  eight lanes instead of one; Tier 1, default on, and the reason that file is not
  bench-only.

Two negative results are kept in the tree opt-in as reproducible references: the
**MoE block dataflow kernel** (`metal/glm53_moe_block.metal`; router, shared expert,
eight routed experts, slot sum, residual add and HC expand in one persistent
240-threadgroup dispatch — bit-exact over 7.3 billion words, 168 fewer dispatches per
token, and slower in the graph, because a 240-threadgroup grid is far more
latency-sensitive than an 8,192-threadgroup one and the counter handoffs cost more than
the ~3 µs of overlap they buy), and **hc_pre algebra half B** (sliced collapse with a
communicated sum of squares and a watchdog-recovered counter wait).

Two hardware facts that shaped the kernels and are worth knowing when reading them: on
the two-die M3 Ultra, plain device stores from one threadgroup are not reliably visible
to an elected last threadgroup in the same dispatch even under seq_cst device-scope
fences — cross-threadgroup data inside one dispatch has to travel through atomics or
cross a dispatch boundary; and fast-math re-associates regardless of source order, so
bit-identity requires identical lane/instruction structure (`acc += x*w` contracts to an
FMA and drifts 1 ULP; `fma(x, w, 0.0f)` pins the product).

## 2. Prefill kernels

Prefill runs different kernels from decode (batched, tiled), so the decode work bought
it nothing; it was taken up separately with the same method (per-stage trace,
`DS4_GLM_PREFILL_TRACE`, `ds4.c:38137`). Default-on unless marked:

- **KDA prefill fast path** (`ds4_metal.m:55510-55660`, `metal/glm53_kda.metal`): a
  prepare/recurrence pair with a 16-column recurrence, staged q/k/decay, and block sizes
  chosen per row count; the sweep knobs that found the shape are still readable from the
  environment.
- **BF16 low-rank split-K** (`ds4_metal.m:52129-52200`, `metal/glm53_bf16.metal`,
  **Tier 2**): the 4-to-8-threadgroup matmul dispatches for the HC mixer, the KDA
  low-rank pair and the DSA indexer k/gate become 16 threadgroups per tile of partial
  sums added in fixed slice order. Its kernel ordering error against an FP64 reference
  is under 1% of the rounding floor main's own input narrowing already introduces.
- **Blocked online softmax in batched DSA attention** (`ds4_metal.m:40862-41100`,
  `metal/dsv4_misc.metal`, **Tier 2**): a 24-row gather stage with a 4-row softmax
  sub-block in one pass; 16-byte gather copies; the rescale skipped where it is an
  identity; the dead rope path removed at this shape.
- **Checked ragged tail** (same kernel, **Tier 2 by registration**, though it corrects
  rather than reorders): upstream's pool expansion writes `0xffffffff` sentinels into
  the 1–3 unused tail slots of three tokens in four, and the valid-only attention kernel
  stages the cache at that row. The branch bounds-checks the tail; exact mode keeps the
  legacy behaviour because upstream has it. `bench/FIDELITY.md` records the measurement.
- **Token-tiled router and qk low-rank kernels** (`ds4_metal.m:24726-24900`, `:39631`):
  one weight pass per block of tokens instead of one per token (the qk low-rank kernel
  had been re-walking a 512×256 Q8_0 matrix per token, 9% of a 62k prefill).
- **Routed-expert GEMM inner loop** (`ds4_metal.m:43790-44020`): 16-byte dequant loads,
  narrow-tile selection, a grouped path above a token threshold; A-stage double
  buffering is implemented but off.
- **Dense half copy / ring** (`ds4_metal.m:21150-21300`) and the **indexer causal grid**
  (`ds4_metal.m:39155`: the staircase of causal scores as a rectangle plus one fill
  kernel instead of a full grid), and the **prefill folds** (width-4 HC expand and
  SwiGLU, per-expert work map, FFN add folded one pass earlier).
- **Prefill chunk** raised to 8192 tokens beyond the dense window
  (`DS4_GLM53_PREFILL_CHUNK`, `ds4.c:37897`), where the chunk only affects buffer sizes
  and routed-tile fill.

The blocked softmax and the split-K were adopted together as one "bundle" against the
1,000-case manifest; that procedure, and why greedy-output identity cannot gate prefill
changes, is in `bench/FIDELITY.md`.

**Long-prompt expert bank.** The integration line can expand the routed Q4_K expert
weights once per layer-major prompt group, reuse that bank across the group's token
tiles, encode the layer in one fused command buffer and pipeline up to eight layers.
On `b723dfa`, the forced configuration reached 550.72 prefill tokens/s at 62,174
prompt tokens and 473.75 at 300,000. The final guarded C defaults restrict automatic
admission to exactly M3 Ultra with at least 500 GiB RAM and the full unsliced, non-SSD,
non-TP 185,299,232,064-byte public Q4_K profile. The final 33,148-token boundary pair
measured 556.73 t/s with AUTO versus 540.64 with the same-binary bank-off control, so
the conservative 32768-token threshold was retained. The ordinary
path remains the refusal/failure fallback, and explicit enable/disable, schedule and
model-mapping controls are documented in `docs/GLM53_M3ULTRA.md`.

The guarded profile also defaults the mmap-backed Metal model view to untracked mode
when `DS4_METAL_MODEL_UNTRACKED` is unset. This decision happens after weight binding,
is independent of the bank kill switch, and does not widen CPU inspection, SSD,
multi-tier or tensor-parallel paths. Explicit `=0` disables it and `=1` enables it on
other compatible Metal profiles. The final 62,174-token receipt resolved
`model_untracked=1` with the guarded bank, fused command buffer and fixed eight-layer
pipeline enabled.

## 3. DFlash2 speculative decoding

`ds4_dflash2.inc`, `ds4_dflash_glm.inc`, `ds4_dflash_seed.inc`, `ds4_dflash_selector.inc`,
`ds4_dflash_golden.inc`, `metal/dflash2.metal`, `gguf-tools/dflash2_to_gguf.py`, and the
`--dflash` flag in `ds4_cli.c:2130` / `ds4_server.c:14488`. A GLM-5.3 port of the
DFlash/DFlash2 block-diffusion draft engine from antirez/ds4 PR #844 (MIT source and
attribution details in
`THIRD_PARTY.md`): a simdgroup SDPA drafter kernel, BF16 drafter weights, a persistent
sliding-window context-KV cache, the z-lab candidate selector, exact rejection sampling
machinery retained for testing, and two greedy public profiles. The upstream drafter is trained
for a fixed anchor-plus-seven block. ds4 preserves that geometry, computes normalized
full-vocabulary confidence for conservative admission, and commits only rows accepted by
the target verifier. Bare startup is serial and does
not load draft weights; `--dflash FILE` defaults to conservative windowed-confidence
admission. Explicit modes win over the legacy environment switches regardless of
argument order.

**Public profiles.** Conservative admits a proposal when the selector's conditional
confidence holds for at least four positions, verifies the full eight-row block, and runs
a windowed cost-feedback controller: three complete attempts are judged together against
the request's measured serial step, a winning window keeps proposing, and a losing window
backs off 16, 32, 64 and at most 128 consumed serial tokens. Requests begin with 16 serial
tokens; reasoning spans decode serially (the server's `<think>` parser feeds
`ds4_session_decode_reasoning`). The earlier per-attempt savings ledger (1% retry tax,
3% soft loss meter, savings-funded retry, early recovery) is retained behind
`DS4_DFLASH_WINDOWED=0`. Speculative is the full-block profile: whenever causal
conditioning and context/response room permit, it drafts and offers all seven positions
without confidence admission, retry, backoff, or a loss meter. The target
still verifies every offered row; history ownership, exact rollback and the session fault
latch still apply. `DS4_DFLASH_BLIND_RESEARCH` is obsolete: speculative now names the
full-block policy directly, and the withdrawn 5% aggressive confidence profile is no longer
public. Full-block speculative can be much slower on bad input. Both verified profiles may
produce a numerically different valid continuation; serial remains the ordinary
serial-arithmetic path.

**Why the controller changed (2026-09-07/08).** The per-attempt ledger was rejected on
the real coding-agent workload: on actual code writes its first verification measured one
draft token, accepted zero, and the extrapolated cost of a seven-token draft was then
above the serial comparison, so the request stayed effectively serial. The windowed
controller, the four-position minimum prefix at raw confidence 0.75, full-block
verification, selector conditional confidence, serial reasoning, the full 2047-row
drafter history and an operator-quantized Q8_0 drafter were selected together on that
agent (32 randomized real requests: 39.73 vs 38.09 output tokens/s, +4.3%, every declined
and failed attempt included; leave-one-request-out stays between +3.2% and +5.6%). Drafter
kernels added for it: split-KV (Flash-Decoding style) draft attention
`kernel_dflash2_sdpa_split`/`_merge` with a stable online-softmax merge and pre-dispatch
fallback, unpadded eight-row draft GEMMs, and vocabulary-head padding for partial-width
verification. Each path is default-on with an `=0` kill; `docs/DFLASH_GLM53.md` sections
7 and 9 record the measurement, its limits, and the Q8_0 recipe
(`gguf-tools/dflash2_quantize_prep.py` + `llama-quantize`). `DS4_DFLASH_STATS=summary`
prints request totals without per-token output. Tests: `tests/test_dflash_windowed.c`,
`tests/test_dflash_selector_confidence.c`, `tests/test_dflash_sdpa.c` (GPU).

**Rollback.** Verification advances both KDA conv/recurrent state and the DSA indexer
tail ring for every offered row. The verifier now captures both state families at each
causal prefix. A partial acceptance restores the accepted-prefix snapshot; a missing or
incomplete snapshot restores the complete pre-block backup and replays the committed
prefix through the ordinary serial forward. `DS4_DFLASH_NO_WIDE_ROLLBACK=1` restores the
previous refusal; `DS4_DFLASH_FORCE_REPLAY=1` selects the replay oracle. Rejected compact
cache rows remain outside the visible frontier until overwritten.

**Greedy only, and request-state ownership.** Positive temperature does not enter the
speculative cycle: its rejection masks the rejected token's logit and every caller then
re-applies top-k/top-p/min-p to that modified vector, so the residual is filtered against
a different support than the target's own distribution and can admit a token the original
filter excluded (`tests/test_dflash_sampling.c` shows it on the real sampler). Separately,
the drafter's context cache, staged features and the process-global prefill seed ring are
now bound to a per-session prefix generation that changes on any non-extending sync, a
restored payload, a rewind or an invalidation — a frontier comparison alone cannot see a
different request of the same length, or a restored prefix longer than the last one. Every
exit after the cycle arms its instrumentation routes through one disarm, and the session's
checkpoint survives such an exit only when the speculative state was actually restored.

**Target compatibility.** The drafter borrows the target's `token_embd` (anchor and mask
rows, read per proposal) and its `output` head. The public target GGUF stores both as
Q8_0, so `dflash2_embed_row` gained a Q8_0 row decode that shares the engine's canonical
Q8_0 decoder, and the target's embedding/head shapes, embedding type, mask token and tap
layers are screened once at load — an unsupported target is refused there with a reason,
not aborted mid-generation. `tests/test_dflash2_embed_q8.c` and
`tests/test_dflash_rollback.c`, `tests/test_dflash_lifecycle.c` and
`tests/test_dflash_sampling.c` cover the row decode, the rollback bookkeeping, the failure
exits and lifecycle identity, and the sampler contract on CPU.
`tests/dflash_rejection_harness.sh` supplies deterministic drafts from a
retained serial continuation so a rejection can be placed after 0, 1, 2 or 3 accepted
drafts on either side of the 4-token pool boundary, and `tests/dflash_cached_depth.c`
measures a restored prefix through the product speculative entry point. The `b723dfa`
affected runtime driver passed 16/16 startup, model-id, cancellation, stop, natural-EOS
and healthy-reuse checks. Its favorable fixed 8,192-token SQL horizon measured
38.5850 / 47.2064 / 60.7875 t/s for serial / conservative / speculative, with all arms
reaching the fixed limit during tuple 483 of a requested 2,000 rather than completing
the task. Those `b723dfa` labels describe that historical runtime and do not certify the
current conservative scheduler or current public full-block implementation. The
personal-use recreation recipe for the drafter — pinned
revision, checksums, converter command, launch flags and smoke tests — is
`docs/DFLASH_GLM53.md`; no drafter weights are redistributed.

Experimental serial-capture epilogue fusion, verifier BF16-MM, verifier indexer and
padded proposer-head paths remain off by default. Their diagnostic force/comparison
switches are not part of either public profile and require their own numerical and
timing evidence before any default change.

## 4. Server

`ds4_server.c`; each item default-on with a switch:

- **Thinking-turn cache keys for GLM** — after a thinking turn the server remembered a
  "visible key" so the next request could continue from live KV, but rendered it in
  DeepSeek syntax; GLM's template keeps the opening `<think>`, trims assistant text and
  has no DeepSeek EOS marker, so on GLM the key never matched and every multi-turn
  thinking exchange with a reasoning-stripping client re-prefilled from scratch (four
  minutes per turn at 100k). Fixed; the checkpoint is now also recorded for turns
  truncated by `max_tokens` and for tool-context turns (`DS4_SERVER_CHECKPOINT_ON_LENGTH`,
  `DS4_SERVER_CHECKPOINT_WITH_TOOLS`, `ds4_server.c:11700-11715`), which had been the
  remaining source of cache misses at depth.
- **KV eviction stores trimmed to the client's transcript** (`ds4_server.c:10431`,
  `DS4_KV_EVICT_RAW` restores the old store): the disk key is otherwise the rendered
  sampled text including hidden thinking and tool markup, which clients that reshape the
  assistant turn never reproduce.
- **Checkpoint granularity** — the recommended `--kv-cache-continued-interval-tokens 4096
  --kv-cache-cold-max-tokens 30000`, after a client inserted a system reminder into the
  middle of a 24k transcript and every strict-prefix cache mechanism missed; a saved
  checkpoint shorter than the divergence point turns a 62 s re-prefill into ~20 s.
- **Streaming guard** (`ds4_server.c:6553`, `DS4_STREAM_GUARD_LEGACY`): with thinking and
  tools both enabled, both emitters held *all* answer text after the first `</think>`
  until a second one, a tool-call start or the end of generation, so the answer arrived
  as one delta. The guard now holds only while the pending bytes could still spell
  `<think>`.
- **Anthropic default effort** (`ds4_server.c:3859`, `DS4_ANTHROPIC_DEFAULT_EFFORT`) for
  clients that cannot send `reasoning_effort`; **slot placement by eviction cost**
  (`ds4_server.c:13682`, only with `--batched-session ≥ 2`, which must not be enabled for
  this workload because batched mode disables speculative decoding); **GLM tool-result
  reorder** (`ds4_server.c:2732`, with a jinja2 reference renderer in
  `tests/glm_tool_result_reorder_ref.py`); **trace segment cap** (`DS4_TRACE_MAX_MB`).
- Two request-surface fixes from the audit: `/v1/models` leads with the loaded model's
  own id (`glm-5.3-flash`, derived from the GGUF architecture) and keeps the GLM-5.2 ids
  as documented aliases; a negative or fractional `max_tokens` is a 400 in the existing
  error shape instead of a 200 with an empty completion (`json_max_tokens()`).
- The merge adopted upstream's newer multimodal session handling (a validity predicate
  instead of the fork's unconditional invalidation); see `docs/GLM53_M3ULTRA.md`.
- **Anthropic content parts are rendered in order** (2026-09-08). A user message can
  interleave `text` blocks with `tool_result` blocks; agent clients do this for
  compaction summaries and follow-up instructions. The GLM renderer treated any message
  carrying a tool result as a sortable tool-only observation block, dropping the text
  order and rendering literal `<tool_result>` text as a tool response. The JSON parser
  now records each part's byte span and type, a mixed message is never sorted, and one
  routine (`append_glm_observation_messages`) renders both full prompts and live
  continuations. `test_glm_mixed_tool_result_instructions` covers it.
- **GLM tool-call maps persist in disk checkpoints** (2026-09-08). The checkpoint
  tool-map writer located tool blocks by the enclosing `tool_calls` tag, which GLM's
  syntax does not emit, so every GLM tool-call ID was silently dropped when a KV
  checkpoint was written and a restart or slot displacement lost the cached prefix. The
  writer now matches the exact bounds the GLM generated-message parser remembers
  (consecutive `tool_call` elements as one block, including two leading newlines and
  trailing whitespace), so the replayed prompt is byte-identical. In the live agent this
  turned a post-compaction re-prefill of a 306k-token transcript into a 1.4 s restore.
  `test_glm_kv_tool_map_roundtrip_exact_blocks` covers eight variants.
- **Images nested in Anthropic `tool_result` content reach the model** (2026-09-10,
  `a0243fb`). A tool result whose content array carries an `image` block (an agent's
  read-file tool returning a PNG or JPEG) was flattened by the generic content parser,
  which keeps only `text` keys, so the image was dropped and the model saw an empty tool
  result and guessed. `json_tool_result_content()` now registers nested image blocks on
  the message and leaves each marker inline, so the image span is rendered inside the
  `<tool_result>`; images collected while parsing a block are attached only when the
  block is a `tool_result`. `test_anthropic_tool_result_image_content` covers
  text/image/text ordering and the part bounds. Verified on the served model: a
  5712×4284 JPEG returned through a `Read` tool is described correctly.
- **The live prefix hit survives an appended image** (2026-09-10, `0d568f8`). The exact
  token-prefix cache hit required the request's image set to equal the checkpoint's, so
  a multimodal conversation that added an image was accounted as a cold prefill:
  `cache_read_input_tokens` 0, the prefill rate line dividing the whole prompt by the
  time spent on the tail (20,000 t/s printed for a 31k prompt whose 645-token tail took
  1.5 s), and monitors deriving impossible rates, while the engine was already reusing
  the common prefix. `ds4_session_vision_prefix_state_matches()` accepts a checkpoint
  whose images match as a prefix when every new image starts at or after the
  checkpoint; the exact-prefix and rewind paths use it, and the text-suffix paths, which
  slice and retokenize, keep the strict match. Verified: a tool-result photo after a
  cached turn reports `cache_read 4,435 / cache_creation 7,938` and the rate line covers
  the computed tail only.
- **Shutdown cancellation across long prefill.** Commit `be75a99` makes the
  non-streaming long-prefill callback observe the process-wide stop state. Shutdown can
  therefore cancel work before generation begins, while teardown still joins worker
  threads before the engine and Metal model views are closed. The focused cancellation
  test is part of the final affected runtime set.

## 5. Exact mode and the fidelity harness

The rule (`bench/FIDELITY.md`): a change either proves bit-identity in a randomized
poisoned harness with a detected control arm (Tier 1), or it is a floating-point-order
change (Tier 2) that ships behind its own kill switch, is registered in
`glm53_exact_mode()` (`ds4_metal.m:52681` and the registry comment after it; mirror
`glm53_exact_mode_c()` in `ds4.c:42329`), and may default on only after the
teacher-forced scorer shows it indistinguishable from the same-build control within a
cumulative 3e-4 avg_nll budget against clean upstream. `DS4_GLM_EXACT=1` clamps all
eleven registered entries at once. Its role is to isolate registered FP-order changes;
it does not undo the corrected DSA pad-row semantics and does not promise universal
byte identity with upstream (`bench/EXACT-MODE-PLAN.md`).

The harness itself: `bench/prefill-gate.sh` (interleaved A/B prefill gate with byte
identity), `bench/tier2-gate.sh` + `bench/compare_1k.py` (the 1,000-case scorer gate
with paired statistics), the two manifest generators and their fixtures, and a set of
in-engine instruments that are off unless asked for — the kernel ledger, prefill
trace, selection trace, top-k and scorer operand capture, an FP64 host reference for
DSA attention, generation counters and an ignore-EOS mode so paired benchmarks get
equal decode windows. Speed-bench microbenches for the small-kernel fusion and the depth
selector have Makefile targets. The tail-sentinel defect in upstream's DSA selection was
found with these instruments.

## 6. Documentation and packaging

`docs/GLM53_M3ULTRA.md` (configuration, contract, feature status with "implemented" and
"validated" kept apart, switch reference, recipes, receipt-bound results, known issues,
checklist); `bench/README.md`, `bench/FIDELITY.md`, `bench/EXACT-MODE-PLAN.md` rewritten
for a public reader with the historical ledger separated from the public-artifact epoch;
the bench scripts parametrized so they run from a fresh checkout; `THIRD_PARTY.md` for
the DFlash2 provenance, the non-redistributable drafter weights and the jinja2 test
dependency; and this file. The campaign's working notes (prefill plan, decode backlog,
exact-reference procedure) are kept outside the public tree; their technical content is
summarized in `bench/README.md` ("Campaign notes").

## How this was built

This was a collaborative agent feedback loop. Fable, working through Claude Code,
implemented the early decode and prefill rounds. Astral, working through Codex,
independently audited the code and sized the next opportunities through shared written
notes, then took implementation, integration and GPU-run ownership when Fable reached
its usage cap. Fable returned as a consulting peer; Astral used Astra and Sol agents for
bounded implementation and review tasks. The agents worked as peers, with Astral acting
as tie breaker only when a disagreement remained unresolved. Mark set the goals and
constraints and steered priorities.

The existing commit authors and contributor trailers preserve the original record; the
curated history does not invent per-commit identities. Performance and fidelity claims
stand on the measured evidence rather than on any contributor's assessment.
