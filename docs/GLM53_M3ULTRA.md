# GLM-5.3-Flash on the Mac Studio M3 Ultra

[README](../README.md) | [Models](MODELS.md) | [Metal](METAL.md) | [Server](SERVER.md)

This branch is a set of Metal decode and prefill kernels, server changes and a
fidelity harness for **single-stream GLM-5.3-Flash on one M3 Ultra**, built on top of
upstream ds4 at commit `9ab7053`. It targets the regime where a coding agent actually
spends its tokens — 40k to 300k+ tokens of context, one request at a time — and it
keeps every registered floating-point-order change behind a switch so those changes
can all be disabled with one environment variable. The measured scope of that
diagnostic is stated below. The engineering
account of what changed and why is in [`CHANGES-GLM53.md`](../CHANGES-GLM53.md); the
fidelity rules are in [`bench/FIDELITY.md`](../bench/FIDELITY.md).

## Tested configuration

| | |
|---|---|
| Machine | Mac Studio M3 Ultra, 512 GB unified memory, macOS 26.6 |
| Backend | Metal (`--metal`); the CUDA/ROCm paths are upstream's and untouched by the kernel work here |
| Model | GLM-5.3-Flash, official zai-org/GLM-5.3-Flash snapshot `84c6a6aa9497188e15a635ba793b0f95a79b1033` |
| Weights | upstream's own Q4 recipe (`gguf-tools/glm53_quantize.py --artifact q4` at `9ab7053`), see below |
| Weights identity | sha256 `828f413cae8ceee74796606814295e00e04c1c7b151aca9a3c5c75db32cb73e0`, 185,299,232,064 bytes |
| Layout | routed experts Q4_K; DSA attention, shared and dense FFN, all KDA projections, `token_embd` and `output.weight` Q8_0; HC and indexer BF16; norms and state F32 |
| Mode | single stream: one CLI session or one server slot, no `--batched-session`, no tensor parallelism |

### Reproducing the weights

The published `glm53-q4` download (`./download_model.sh glm53-q4`) is a different
conversion of the same snapshot. The file this branch was validated on was produced with
upstream's converter at the same commit as the branch base, with no importance matrix
(upstream ships none for GLM-5.3), using the published Q4_K file only as the source of
tokenizer metadata:

```sh
# official FP8 snapshot, pinned revision (62 safetensors shards, ~328 GB)
huggingface-cli download zai-org/GLM-5.3-Flash \
    --revision 84c6a6aa9497188e15a635ba793b0f95a79b1033 --local-dir hf/GLM-5.3-Flash
./download_model.sh glm53-q4            # tokenizer template only
make -C gguf-tools                      # builds libds4quants
python3 gguf-tools/glm53_quantize.py \
    --hf hf/GLM-5.3-Flash \
    --tokenizer-template gguf/GLM-5.3-Flash-Q4_K.gguf \
    --artifact q4 --threads 16 \
    --source-revision 84c6a6aa9497188e15a635ba793b0f95a79b1033 \
    --out gguf/GLM-5.3-Flash-Q4_K.gguf
shasum -a 256 gguf/GLM-5.3-Flash-Q4_K.gguf   # expect 828f413c…
```

The converter validates the index, every shard header and the FP8 scales before
writing, and checks template-token equality against the snapshot's tokenizer.

## Build and run

```sh
make ds4 ds4-server -j12
make -C gguf-tools quality-score        # the teacher-forced scorer (optional)

# CLI, greedy, 70k context
DS4_GLM53_MEMORY_CEILING_GB=280 \
  ./ds4 -m gguf/GLM-5.3-Flash-Q4_K.gguf --metal -c 70000 --nothink --temp 0 \
        -n 2048 --prompt-file prompt.txt

# Server, as deployed for coding agents: disk KV checkpoints, fine-grained
# continued checkpoints so a mid-transcript edit costs a partial re-prefill
DS4_GLM53_MEMORY_CEILING_GB=280 DS4_ANTHROPIC_DEFAULT_EFFORT=max \
  ./ds4-server -m gguf/GLM-5.3-Flash-Q4_K.gguf --metal \
        --host 127.0.0.1 --port 8095 -c 409600 \
        --kv-disk-dir kv --kv-disk-space-mb 98304 \
        --kv-cache-continued-interval-tokens 4096 --kv-cache-cold-max-tokens 30000
```

`ds4` compiles its Metal library from `metal/*.metal` relative to the current directory,
so run the binaries from the repository root (or the tree they were built in).
`DS4_GLM53_MEMORY_CEILING_GB` is the only environment variable most users will want: it
clamps the model's memory guard so other work on a 512 GB machine stays safe (280 GB
leaves room for a 400k context; 400 GB is used for scorer runs).

Kernel defaults are the "fast mode" of `bench/FIDELITY.md`: every adopted kernel
change on, including the eleven registered floating-point-order changes. The final
release also plans a guarded M3 Ultra/public-Q4 mapping and expert-bank profile described
below. That policy is still pending its final merge and screen, so `b723dfa` required
explicit bank and mapping controls for its capability receipts.

### Model ids

`GET /v1/models` lists the loaded model under its own id first. The id is derived from
the GGUF at load: `general.architecture = glm5-next` selects the GLM-5.3 shape
(`ds4.c`, `config_validate_model`), and `server_model_id_from_engine()`
(`ds4_server.c`) maps that to `glm-5.3-flash`, the same id a request that names no model
is assigned. The listing is, in order:

| id | effect |
|---|---|
| `glm-5.3-flash` | the model; thinking controlled by the request |
| `glm-5.3-flash-chat` (also `-no-think`, `-nothink`) | thinking off |
| `glm-5.3-flash-reasoner` | thinking on |
| `glm-5.2`, `glm-5.2-chat`, `glm-5.2-reasoner` | **legacy aliases** kept for clients configured against the GLM-5.2 ids; same model, same thinking semantics as the `glm-5.3-flash` forms |

`zai/`-prefixed forms of all of the above are accepted in requests and on
`GET /v1/models/<id>` but are not listed. The `name` field of every entry is the loaded
shape's name (`GLM 5.3 Flash`).

## Exact-mode diagnostic

`DS4_GLM_EXACT=1` turns every registered floating-point-order change off. It is a
diagnostic with a measured scope, not a promise of byte identity with upstream. On
three diagnostic long-context cases, exact-mode total NLL differed from upstream by
0.000238, 0.004171 and 0.000048. The corrected DSA pad-row semantics stay active under
exact mode and are not expected to reproduce upstream's legacy pad-row behavior. The
switch clamps the eleven registry entries listed
in [`bench/EXACT-MODE-PLAN.md`](../bench/EXACT-MODE-PLAN.md) §1a (hc_pre slice count,
BF16 low-rank split-K, blocked DSA softmax, the checked DSA tail, the `xr8` scorer, the
hc_pre algebra halves, split8 block rows, the Q8_0 and BF16 matvec simdgroup counts, and
the two split8 opt-ins), and by dispatch policy it also selects the production
HC-expand epilogue instead of `ptail`.

In plain words, the two tiers of `bench/FIDELITY.md`:

- **Tier 1** — the same arithmetic in the same order; only how the work is laid out on
  the GPU changes (fusing dispatches, widening threadgroups, changing which lane
  computes what). These are proven bit-identical in a randomized harness (≥ 50,000
  poisoned draws, every output word compared, 1,000-run determinism) and need no exact-
  mode entry; each keeps a `DS4_GLM_DISABLE_*` switch for A/B measurement.
- **Tier 2** — a different summation order (split-K partial sums, a different reduction
  tree). These may move the last bits. Each one ships behind its own kill switch *and*
  is registered so `DS4_GLM_EXACT=1` turns it off, and it may default on only after the
  teacher-forced scorer shows it statistically indistinguishable from the same-build
  control and the whole set stays within 3e-4 avg_nll of clean upstream.

The three-way diagnostic against an upstream pin, the same-build all-off arm and fast
mode is in `bench/EXACT-MODE-PLAN.md` §2. It records numerical differences rather than
assuming the upstream TSV must be byte-identical.

## Feature status

"Implemented / default" is what the code does; "validated on the public artifact" is
what has been run, on the merged branch, against the weights above. The two columns
are deliberately separate.

| feature | implemented / default | validated on the merged branch with the public artifact |
|---|---|---|
| Decode kernel set (HC-pre single dispatch and wide tail, hc-mix split-K, KDA low-rank pack/fold and glue, DSA indexer fold and pair, routed-down split, router shared fold and tail fold, top-k radix fast path, epilogue fusions) | implemented, default on, each with a kill switch | `b723dfa`: complete native 300k serial block at 37.2994 t/s; final rebuilt release receipt pending |
| Prefill kernel set (KDA prefill fast path, BF16 low-rank split-K, blocked DSA softmax, checked DSA tail, token-tiled router and qk low-rank, dense half copy/ring, indexer causal grid, prefill folds) | implemented, default on, each with a kill switch | `b723dfa`: 550.72 t/s at 62,174 prompt tokens and 473.75 t/s at 300,000; final rebuilt release receipt pending |
| Long-prompt expert bank | implemented on the integration line; guarded defaults are being merged after `b723dfa` | intended auto profile is exactly M3 Ultra with at least 500 GiB RAM and the full unsliced, non-SSD, non-TP 185,299,232,064-byte Q4_K artifact; `b723dfa` capability receipts used forced bank controls, so the final default build and 32k boundary still need screening |
| `ptail` HC-expand epilogue (kernel in `metal/t2screen.metal`) | implemented, default on (`DS4_GLM_DISABLE_HCX_PTAIL=1` off) | Tier 1 receipt on the pre-merge tree; dispatch confirmed in the E1 log (`T2SCREEN first-dispatch HCXTAIL`) |
| `DS4_GLM_EXACT` umbrella, 11 registry entries | implemented | three long-context diagnostics within 0.0042 total NLL per case; full public-artifact diagnostic remains pending |
| MoE block dataflow kernel, hc_pre algebra half B, one-dispatch hc_pre, `xr8` scorer, split8 opt-ins | implemented, **opt-in** (measured slower, or unexplained divergence in the case of `xr8`) | not part of the validated defaults |
| All-Q8 KDA projection fusion (`DS4_GLM_ENABLE_KDA_PROJ_FUSE`) | implemented, opt-in; the first path written for this file's all-Q8 KDA layout | **not validated**; needs an identity check against the default before it can be recommended |
| Server: Anthropic default effort, KV checkpoint / eviction policy, streaming guard, slot scoring, GLM tool-result reorder | implemented, default on | affected runtime driver on `b723dfa`: 16/16 checks passed, including model aliases, explicit DFlash mode precedence, bank cancellation and healthy reuse, speculative stop and natural EOS; final rebuilt receipt pending |
| Multimodal (vision) requests | upstream's newer behaviour (session reused when the vision state matches) adopted in the merge | **re-validation pending** on a vision prompt |
| MTP row-boundary KDA snapshot (`--mtp` reject-replay fast path) | compiled but **inert** on real GLM-5.3 graphs: guarded so it fires only when the snapshot covers the whole speculative state (`ds4.c:68578`); otherwise upstream's full restore+replay runs | n/a — the guard makes the path equivalent to upstream's |
| DFlash2 speculative decoding (`--dflash`) | optional and greedy-only. Bare startup is serial; a supplied drafter defaults to conservative request-credit scheduling; `--dflash-mode speculative` selects the uncapped policy. Positive temperature decodes serially. | `b723dfa` passed the affected runtime suite. On one favorable fixed 8,192-token SQL horizon: serial 38.5850, conservative 47.2064, speculative 60.7875 t/s; final rebuilt receipt pending. |
| CUDA / ROCm / tensor parallel / SSD streaming | upstream's, plus small GLM-5.3 additions in `ds4_cuda.cu` and `rocm/ds4_rocm_glm.cuh` (see "Dispositions") | not built or run on this branch |

**How DFlash2's rollback was completed.** Verification snapshots both the KDA
state and the DSA indexer's incomplete key/gate tails at each block position.
On partial acceptance, it restores the accepted prefix instead of replaying all
accepted tokens through the model. The complete pre-block backup remains the
fallback when capture coverage is incomplete; `DS4_DFLASH_FORCE_REPLAY=1`
selects the serial replay reference, including fully accepted blocks.

Focused tests compare complete recurrent state, live caches and continued logits
while changing rejected tokens, including the first rejected row and a pooled-key
boundary. These tests establish causality for the tested frontiers. They do not
promise byte equality between all batched and serial arithmetic. See
`tests/DFLASH-PREFIX.md` for the test scope and reproduction commands. The final
combined branch still needs its affected correctness, serving and performance
checks; the adaptive admission policy remains under development.

### Dispositions

Stated once, so a reader does not have to infer them from the table:

- **DFlash2 speculative decoding is optional and greedy-only.** A bare startup is
  serial and allocates no drafter state. Supplying `--dflash FILE` defaults to the
  conservative request-credit policy; speculative selects the uncapped policy and can
  regress. Both retain the target model as verifier and are intended to preserve model
  quality. The conservative policy has an empirical less-than-2% added-work target on
  its calibrated M3 Ultra profile, not a universal physical guarantee. Personal-use
  draft recreation instructions are in `docs/DFLASH_GLM53.md`; no draft weights ship.
- **Unsupported or untested configurations.** The CUDA and ROCm paths carry small
  GLM-5.3 additions (`ds4_cuda.cu`, `rocm/ds4_rocm_glm.cuh`) that are compile-only as far
  as this branch goes: no CUDA or ROCm machine built or ran them here; tensor
  parallelism and SSD streaming are upstream's and not exercised; the multimodal
  (vision) session reuse adopted in the merge has not been re-validated on this branch.
  None of these should be assumed to work here beyond what upstream `9ab7053` already
  established.
- **Cache and recovery cases not validated here.** The lifecycle suite covers a
  same-process sequence (cold prefill, one snapshot stored, an immediate repeat,
  disconnects, invalid requests, over-context prompts). It does not exercise a server
  restart against a warm disk cache, a session reset, or eviction under disk-space
  pressure; those paths are upstream's and remain **not validated here**.

### Guarded expert-bank defaults pending final integration

The planned C defaults enable the expert bank only when all of these checks pass:
Metal reports exactly Apple M3 Ultra, physical memory is at least 500 GiB, and the
model is the full unsliced, non-SSD, non-TP public GLM-5.3 Q4_K profile with the
185,299,232,064-byte artifact shape and all 42 routed-expert tensors in Q4_K. The
automatic schedule uses the bank, fused layer command buffer and an eight-layer
pipeline only for prompts at or above `DS4_GLM_EXPERT_BANK_MIN_TOKENS` (planned
default 32768). That threshold is a conservative engineering boundary pending the
final 32k screen, not a measured crossover.

`DS4_GLM_ENABLE_EXPERT_BANK=1` force-enables another compatible profile; `=0` forces
it off. Forcing the bank on an unmatched profile does not also force the fused command
buffer or pipeline: enable those controls explicitly. `DS4_GLM_DISABLE_EXPERT_BANK=1`
is the strong kill switch. The fused and pipelined controls accept `1` or `0`, with
unset taking the guarded profile default. `DS4_GLM_EXPERT_BANK_PIPELINED_LAYERS` is a
boolean; `=1` selects the fixed eight-layer bound rather than accepting a layer count,
and it requires the fused layer command buffer. `DS4_GLM_DISABLE_SUPERCHUNK=1` is the
strong schedule kill. Dynamic memory admission and ordinary fallback still apply.
These defaults are not present at the `b723dfa` capability commit: release
documentation must not call them shipped until their implementation, final build and
boundary screen are complete.

Speculative DFlash prefill seeding captures tap-layer activations inside the ordinary
per-chunk schedule, so an armed speculative seed makes the expert-bank superchunk
refuse. Conservative mode does no speculative prefill capture and remains eligible for
the bank; serial mode loads no drafter and is eligible as well.

For the same validated profile, the planned Metal mapping default also enables
untracked model views when `DS4_METAL_MODEL_UNTRACKED` is unset. `=0` disables that
mapping explicitly and `=1` enables it elsewhere. The mapping decision is independent
of the bank kill switch, so an auto-vs-disabled bank comparison keeps the model view
constant. CPU inspection, SSD streaming, tensor parallelism and other profiles do not
inherit this automatic mapping. Startup prints the resolved `model_untracked=0|1`.

## Environment switches introduced by this branch

Derived from the `"DS4_*"` string literals read by `getenv` and the engine's env helpers
in `ds4.c`, `ds4_metal.m`, `ds4_cli.c`, `ds4_server.c`, `ds4_gpu.h` and the
`ds4_dflash*.inc` files, minus those present in upstream `9ab7053`: 186 names plus one name family
(`DS4_GLM_T2S_Q8NSG_<FAM>`). One switch named in a source comment,
`DS4_GLM_DISABLE_HC_PRE_SLICES`, is not read anywhere and is omitted. Classes:

- **supported control** — a user-facing setting with a documented default;
- **kill switch** — turns a default-on change off; exists for A/B measurement and
  bisection, and is what `DS4_GLM_EXACT` uses for the registered Tier 2 items;
- **experimental opt-in** — implemented and kept, off by default because it measured
  slower or is not certified;
- **developer instrumentation (bench-only)** — traces, captures, counters, sweep knobs
  and harness aids; never needed to run the model, and candidates for removal or a
  build flag.

Unless a meaning says otherwise, a switch is read as "set to any non-empty value" and
`=0` is not special; the `DS4_GLM_ENABLE_*` controls that accept `=0` say so. The
`DS4_DFLASH_*` switches take effect on this branch now that DFlash2 runs; `--dflash` is
opt-in, so they matter only when a drafter is loaded.

| switch | class | meaning | site |
|---|---|---|---|
| `DS4_ANTHROPIC_DEFAULT_EFFORT` | supported control | Default reasoning effort for Anthropic-protocol requests that carry none (e.g. Claude Code); explicit request fields still win. | `ds4_server.c:3859` |
| `DS4_DFLASH_CTX_CAP` | supported control | Caps the drafter's context rows (default about 256; more rows cost draft latency). | `ds4_dflash_glm.inc:174`, `ds4_dflash_seed.inc:31` |
| `DS4_DFLASH_DISABLE` | supported control | With no explicit `--dflash-mode`, selects serial startup and avoids loading the drafter. An explicit mode wins. | `ds4.c`, DFlash mode resolver |
| `DS4_GLM53_MEMORY_CEILING_GB` | supported control | Clamps the GLM-5.3 memory-guard budget to N GB (used to keep a 512 GB machine's other workloads safe). | `ds4.c:42037` |
| `DS4_GLM53_PREFILL_CHUNK` | supported control | Upper bound on prefill chunk tokens (default 8192; 4096 and 2048 restore earlier shipped chunks). | `ds4.c:37902` |
| `DS4_GLM_ENABLE_EXPERT_BANK` | supported control (pending guarded default) | Unset selects the exact M3 Ultra/public-Q4 profile automatically; `1` forces another compatible profile; `0` forces off. Final implementation and screen pending. | expert-bank admission |
| `DS4_GLM_EXPERT_BANK_MIN_TOKENS` | supported control (pending guarded default) | Minimum prompt tokens for the automatic bank schedule; planned default 32768, a provisional engineering boundary pending the final 32k screen. | expert-bank admission |
| `DS4_GLM_EXPERT_BANK_FUSED_LAYER_CB` | supported control (pending guarded default) | Unset takes the guarded profile default; `1` enables and `0` disables the fused layer command buffer. | expert-bank scheduler |
| `DS4_GLM_EXPERT_BANK_PIPELINED_LAYERS` | supported control (pending guarded default) | Boolean: unset takes the guarded profile default, `1` enables the fixed eight-layer pipeline, and `0` disables it. It is not a layer-count setting and requires the fused layer command buffer. | expert-bank scheduler |
| `DS4_METAL_MODEL_UNTRACKED` | supported control (pending guarded default) | Unset enables untracked model views only on the validated Metal M3 Ultra/public-Q4 profile; `1` enables elsewhere and `0` disables explicitly. Independent of bank admission. | Metal model mapping |
| `DS4_GLM_DSA_TAIL_CHECKED` | supported control | =0 restores the legacy unchecked ragged tail in DSA attention (default 1: bounds-checked; registry entry 4). | `ds4_metal.m:41060` |
| `DS4_GLM_ENABLE_BF16_LOWRANK_SPLITK` | supported control | =0 turns the BF16 low-rank split-K off (default on since bundle round 1). | `ds4_metal.m:52171` |
| `DS4_GLM_ENABLE_DSA_BLOCKED_SOFTMAX` | supported control | =1 forces the blocked softmax on (default on). | `ds4_metal.m:41029` |
| `DS4_GLM_ENABLE_HCX_PTAIL` | supported control | =1 explicitly selects the ptail HC-expand epilogue (already the default). | `ds4_metal.m:51815` |
| `DS4_GLM_ENABLE_ROUTED_DOWN_SPLIT` | supported control | =1 forces the expert-parallel routed down split on (default on). | `ds4_metal.m:42895` |
| `DS4_GLM_ENABLE_ROUTER_SHARED_FOLD` | supported control | =1 forces the router/shared-expert fold on (it is the default; the kill switch wins). | `ds4_metal.m:42127` |
| `DS4_GLM_ENABLE_TOPK_FAST` | supported control | No-op kept for scripts that set it (the fast path is on by default). | `ds4_metal.m:20439` |
| `DS4_GLM_EXACT` | supported control | =1 diagnostic: every registered floating-point-order change off. It does not undo the corrected DSA pad-row semantics and is not a universal upstream-byte-identity promise (see EXACT-MODE-PLAN.md). | `ds4.c:42332` |
| `DS4_GLM_HC_PRE_ALGEBRA_A` | supported control | =0 turns off hc_pre algebra half A (unscaled split-K dot, scale in the tail; default on, registry entry 6). | `ds4_metal.m:53058` |
| `DS4_GLM_HC_PRE_SLICES` | supported control | =8|16|32 hc_pre split-K slice count (default 16; 32 is faster but out of the fidelity budget). | `ds4_metal.m:52827` |
| `DS4_GLM_SCORER_XREDUCE_MIN_ROWS` | supported control | Candidate-row threshold below which the production scorer is kept even when xr8 is enabled (default 37500; 0 disables the gate). | `ds4_metal.m:20333` |
| `DS4_GLM_SPLIT8_BLOCK_ROWS_DEEP` | supported control | Block rows for split8 DSA attention at depth (default 128; registry entry 7). | `ds4.c:42367` |
| `DS4_GLM_SPLIT8_BLOCK_ROWS_SHALLOW` | supported control | Block rows for split8 DSA attention at shallow depth (default 32; registry entry 7). | `ds4.c:42366` |
| `DS4_GLM_T2S_BF16_NSG` | supported control | Simdgroups per threadgroup of the BF16 matvec (registry entry 9; pinned under DS4_GLM_EXACT). | `ds4_metal.m:52047` |
| `DS4_GLM_T2S_Q8NSG_<FAM>` | supported control | Per-family override of the Q8_0 matvec simdgroup count (families KDA, HCX, SDN, SHG, DSAF; registry entry 8; pinned under DS4_GLM_EXACT). | `ds4_metal.m:6303` |
| `DS4_GLM_TOPK_FAST_MIN_COMP` | supported control | Minimum candidate width for the top-k fast path (default 12288; 0 removes the gate). | `ds4_metal.m:20468` |
| `DS4_SERVER_CHECKPOINT_ON_LENGTH` | supported control | =0 stops recording the thinking checkpoint for turns truncated by max_tokens (default: recorded, so the next turn continues from live KV). | `ds4_server.c:11713` |
| `DS4_SERVER_CHECKPOINT_WITH_TOOLS` | supported control | =0 stops recording the thinking checkpoint for tool-context turns (default: recorded). | `ds4_server.c:11710` |
| `DS4_TRACE_MAX_MB` | supported control | Caps the live --trace segment at N MB; the previous segment is kept at <path>.1. | `ds4_server.c:11060` |
| `DS4_DFLASH_NO_ADAPTIVE` | kill switch | Disables the adaptive break-even throttle that parks speculation when it cannot pay. | `ds4_dflash_glm.inc:224` |
| `DS4_DFLASH_NO_WIDE_ROLLBACK` | kill switch | Refuses the speculative cycle on any graph whose speculative state includes the DSA indexer tail (every real GLM-5.3 graph), before any target state is mutated, and decodes serially. Restores this branch's previous behaviour. | `ds4_dflash_glm.inc:88` |
| `DS4_DFLASH_FORCE_REPLAY` | developer instrumentation (bench-only) | Takes the restore-and-replay rollback even where the per-step snapshot is complete; the correctness oracle for A/B. | `ds4_dflash_glm.inc:89` |
| `DS4_DFLASH_SCRIPT` | developer instrumentation (bench-only) | Deterministic draft supply `P:K:N,...`: at position P propose N drafts of which the first K are the retained serial continuation and row K is a known-wrong token; unnamed positions decode serially. The verifier is untouched. | `ds4_dflash_script.inc` |
| `DS4_DFLASH_SCRIPT_IDS` | developer instrumentation (bench-only) | The retained serial continuation the script proposes from. | `ds4_dflash_script.inc` |
| `DS4_DFLASH_SCRIPT_IDS_OUT` | developer instrumentation (bench-only) | Control arm: record the continuation as "position token" lines. | `ds4_dflash_script.inc` |
| `DS4_DFLASH_SCRIPT_SERIAL` | developer instrumentation (bench-only) | Control arm: decode every position serially through the same binary and dumps. | `ds4_dflash_script.inc` |
| `DS4_DFLASH_SCRIPT_DUMP` | developer instrumentation (bench-only) | Directory for frontier logits and per-tensor digests of the complete speculative state. | `ds4_dflash_script.inc` |
| `DS4_DFLASH_FAIL` | developer instrumentation (bench-only) | Inject a cycle failure at `state_save`, `after_arm` or `after_verify` to exercise the cleanup on those exits. | `ds4_dflash2.inc` |
| `DS4_DFLASH_NO_SELECTOR` | kill switch | Disables the DFlash2 candidate selector (coherent-chain tracing). | `ds4_dflash_selector.inc:29` |
| `DS4_DFLASH_SDPA_SCALAR` | kill switch | Forces the scalar SDPA drafter kernel instead of the simdgroup one. | `ds4_metal.m:56113` |
| `DS4_GLM_DISABLE_EXPERT_BANK` | kill switch (pending guarded default) | `=1` strongly disables expert-bank admission; `=0` is a no-op. | expert-bank admission |
| `DS4_GLM_DISABLE_SUPERCHUNK` | kill switch (pending guarded default) | `=1` strongly disables the bank superchunk schedule; `=0` is a no-op. | expert-bank scheduler |
| `DS4_GLM_DISABLE_BF16_LOWRANK_SPLITK` | kill switch | Ordinary mm kernel for the BF16 low-rank prefill matmuls instead of split-K (registry entry 2). | `ds4_metal.m:52173` |
| `DS4_GLM_DISABLE_DENSE_HALF_COPY` | kill switch | Disables the dense-layer half-copy path and its ring (prefill lever 24). | `ds4_metal.m:21155` |
| `DS4_GLM_DISABLE_DENSE_HALF_RING` | kill switch | Disables the half-copy ring alone. | `ds4_metal.m:21299` |
| `DS4_GLM_DISABLE_DSA_BATCH_HOIST` | kill switch | Batched DSA prefill attention: no hoisting of the shared loads. | `ds4_metal.m:40884` |
| `DS4_GLM_DISABLE_DSA_BATCH_NOROPE` | kill switch | Batched DSA prefill attention: keep the (dead at this shape) rope path. | `ds4_metal.m:40885` |
| `DS4_GLM_DISABLE_DSA_BATCH_SKIP_RESCALE` | kill switch | Batched DSA prefill attention: no rescale skipping. | `ds4_metal.m:40883` |
| `DS4_GLM_DISABLE_DSA_BLOCKED_SOFTMAX` | kill switch | Row-at-a-time softmax in batched DSA attention instead of the blocked online softmax (registry entry 3). | `ds4_metal.m:41030` |
| `DS4_GLM_DISABLE_DSA_GATHER_WIDE` | kill switch | 8-byte gather copies in the blocked DSA kernel instead of 16-byte (bit-identical). | `ds4_metal.m:41050` |
| `DS4_GLM_DISABLE_DSA_INDEXER_FOLD` | kill switch | Separate dispatches for q_a, kv_a and the three DSA indexer projections instead of one (bit-identical fold). | `ds4.c:44603` |
| `DS4_GLM_DISABLE_DSA_TAIL_CHECKED` | kill switch | Kill switch for the checked ragged tail. | `ds4_metal.m:41062` |
| `DS4_GLM_DISABLE_HCX_PTAIL` | kill switch | Production HC-expand epilogue instead of the eight-lane ptail epilogue (Tier 1 default; DS4_GLM_EXACT also selects production). | `ds4_metal.m:51816` |
| `DS4_GLM_DISABLE_HC_ATTN_TAILFUSE` | kill switch | Residual add + HC expand as separate dispatches after the attention-output matvec. | `ds4.c:45238` |
| `DS4_GLM_DISABLE_HC_CHAIN_COLLAPSE` | kill switch | Prefill HC chain: separate collapse and weighted RMSNorm passes. | `ds4_metal.m:3408` |
| `DS4_GLM_DISABLE_HC_CHAIN_SCALE` | kill switch | Prefill HC chain: the RMSNorm materializes its own scaled copy instead of the expand publishing the per-row scale. | `ds4_metal.m:3403` |
| `DS4_GLM_DISABLE_HC_MIX_SPLITK` | kill switch | Row-per-simdgroup HC mixer instead of the split-K mixer (historical FP-order change; scorer-identical at the time). | `ds4.c:45005` |
| `DS4_GLM_DISABLE_HC_NORM_FUSE` | kill switch | Unfused HC norm dispatches (from the DSV4 decode ladder). | `ds4.c:44984` |
| `DS4_GLM_DISABLE_HC_NORM_MIX_FUSE` | kill switch | Unfused HC norm+mix dispatches. | `ds4.c:44979` |
| `DS4_GLM_DISABLE_HC_PHASEC_HOIST` | kill switch | Kill switch for the phase-C hoist experiment. | `ds4_metal.m:53093` |
| `DS4_GLM_DISABLE_HC_PRE_ALGEBRA_A` | kill switch | Kill switch for algebra half A. | `ds4_metal.m:53060` |
| `DS4_GLM_DISABLE_HC_PRE_ALGEBRA_B` | kill switch | Kill switch for algebra half B. | `ds4_metal.m:53061` |
| `DS4_GLM_DISABLE_HC_PRE_ONE_DISPATCH` | kill switch | Kill switch for the one-dispatch hc_pre experiment. | `ds4.c:45039` |
| `DS4_GLM_DISABLE_HC_PRE_SINGLE` | kill switch | Unfused hc_pre ladder instead of the single-dispatch fused kernel. | `ds4.c:45010` |
| `DS4_GLM_DISABLE_HC_PRE_WIDE` | kill switch | hc_pre split-K at the decode default slice count instead of 16 (exact-mode registry entry 1). | `ds4_metal.m:52832` |
| `DS4_GLM_DISABLE_HC_REFUSE` | kill switch | Four-dispatch hc_pre instead of the fused rms+split-K / reduce+wsum pair. | `ds4.c:45029` |
| `DS4_GLM_DISABLE_HC_TAILFUSE` | kill switch | Residual add + HC expand as separate dispatches after the MoE shared-down / dense-down matvec. | `ds4.c:46471` |
| `DS4_GLM_DISABLE_HC_TAIL_W4` | kill switch | Scalar split-K reduce in the hc_pre tail instead of float4. | `ds4_metal.m:53102` |
| `DS4_GLM_DISABLE_HC_TAIL_WIDE` | kill switch | Single 1024-thread hc_pre tail instead of the replicated wide tail. | `ds4_metal.m:53098` |
| `DS4_GLM_DISABLE_INDEXED_SPLIT8` | kill switch | Mono DSA decode attention kernel for every n_selected instead of the split8 kernel above 64 rows. | `ds4.c:42383` |
| `DS4_GLM_DISABLE_INDEXER_CAUSAL_GRID` | kill switch | Full rectangular indexer score grid in prefill instead of the causal staircase plus fill kernel. | `ds4_metal.m:39155` |
| `DS4_GLM_DISABLE_INDEXER_HEADFOLD` | kill switch | Disables the indexer head-fold variant selection (prefill indexer projections). | `ds4_metal.m:709` |
| `DS4_GLM_DISABLE_INDEXER_PAIR` | kill switch | Separate dispatches for the indexer k projection and compressor gate. | `ds4.c:44590` |
| `DS4_GLM_DISABLE_INDEXER_SCORE_STREAM` | kill switch | Non-streamed DSA indexer scorer (query re-read per threadgroup). | `ds4_metal.m:20122` |
| `DS4_GLM_DISABLE_KDA_LOWRANK_PACK` | kill switch | Undoes the KDA low-rank pack (f_a+g_a+beta flat3, f_b+g_b pair2in; de-aliased kda_lowrank2). | `ds4.c:45375` |
| `DS4_GLM_DISABLE_KDA_LOWRANK_PROLOGUE` | kill switch | f_b/g_b as separate dispatches instead of in prep_state's per-head prologue. | `ds4.c:45471` |
| `DS4_GLM_DISABLE_KDA_OUT_FOLD` | kill switch | KDA decode_out as its own dispatch instead of folded into the state kernel's tail. | `ds4.c:45473` |
| `DS4_GLM_DISABLE_KDA_PREFILL_C16` | kill switch | KDA prefill at the C=8 configuration instead of the 16-column recurrence. | `ds4_metal.m:55532` |
| `DS4_GLM_DISABLE_KDA_PREFILL_FAST` | kill switch | Production three-kernel KDA prefill instead of the fast path. | `ds4_metal.m:55523` |
| `DS4_GLM_DISABLE_KDA_QKV_LOWRANK_FOLD` | kill switch | Separate dispatch for the three low-rank rows instead of riding in the q/k/v grid's tail. | `ds4.c:45374` |
| `DS4_GLM_DISABLE_KDA_REFUSE` | kill switch | Three-dispatch KDA decode (prep / state / out) instead of the two-dispatch fused prep+state. | `ds4.c:43676` |
| `DS4_GLM_DISABLE_KDA_SPLIT` | kill switch | Single fused KDA decode kernel instead of the split form (falls back one level further). | `ds4.c:43661` |
| `DS4_GLM_DISABLE_LAZY_BATCH_WS` | kill switch | =1 allocates every batch workspace tensor eagerly instead of on first use. | `ds4.c:43708` |
| `DS4_GLM_DISABLE_MOE_BLOCK_DATAFLOW` | kill switch | Kill switch for the MoE block dataflow kernel. | `ds4_metal.m:42379` |
| `DS4_GLM_DISABLE_PREFILL_FOLD_FFNADD` | kill switch | Prefill: FFN residual add as its own pass instead of folded (bit-exact). | `ds4.c:48463` |
| `DS4_GLM_DISABLE_PREFILL_FOLD_HCEXPAND` | kill switch | Prefill: width-1 HC expand instead of width-4. | `ds4_metal.m:3363` |
| `DS4_GLM_DISABLE_PREFILL_FOLD_MOEMAP` | kill switch | Prefill: single-threadgroup routed work map instead of per-expert threadgroups. | `ds4_metal.m:3381` |
| `DS4_GLM_DISABLE_PREFILL_FOLD_MOESWIGLU` | kill switch | Prefill: width-1 SwiGLU passes instead of width-4. | `ds4_metal.m:3369` |
| `DS4_GLM_DISABLE_QAKV_FUSE` | kill switch | Separate dispatches for attn q_a and kv_a. | `ds4.c:44626` |
| `DS4_GLM_DISABLE_QKLOW_BATCH_TILE` | kill switch | Per-(head, token) qk low-rank prefill kernel instead of the token-tiled one. | `ds4_metal.m:39631` |
| `DS4_GLM_DISABLE_QKLOW_SG` | kill switch | Thread-per-row qk low-rank decode kernel instead of the coalesced simdgroup kernel at qk_nope=256. | `ds4_metal.m:39528` |
| `DS4_GLM_DISABLE_REDUCE_Q8_U16` | kill switch | Byte-wise Q8_0 value-row loads in the split8 reduce instead of ushort pairs. | `ds4_metal.m:40315` |
| `DS4_GLM_DISABLE_REDUCE_WIDE_TG` | kill switch | 256-thread split8 reduce dispatch instead of 512. | `ds4_metal.m:40136` |
| `DS4_GLM_DISABLE_ROUTED_ASTAGE_DB` | kill switch | Disables A-stage double buffering in the routed-expert prefill GEMM (default off at compile time; see DS4_GLM_ROUTED_ASTAGE_DB). | `ds4_metal.m:43821` |
| `DS4_GLM_DISABLE_ROUTED_DEQ_WIDE` | kill switch | Scalar quant-byte loads in the routed prefill GEMM dequant instead of 16-byte loads. | `ds4_metal.m:43834` |
| `DS4_GLM_DISABLE_ROUTED_DOWN_SPLIT` | kill switch | Sequential routed-expert down matvec instead of the expert-parallel split (Tier 2; also off under DS4_GLM_EXACT). | `ds4_metal.m:42893` |
| `DS4_GLM_DISABLE_ROUTED_GATEUP_WIDE` | kill switch | Production routed gate+up kernel instead of the ushort4-load variant. | `ds4_metal.m:42932` |
| `DS4_GLM_DISABLE_ROUTED_TILE_NARROW` | kill switch | Plain 32-row routed prefill tile instead of the narrow tile selection. | `ds4_metal.m:43852` |
| `DS4_GLM_DISABLE_ROUTER_BATCH_TILE` | kill switch | Per-token router matvec in prefill instead of the token-tiled kernel. | `ds4_metal.m:24726` |
| `DS4_GLM_DISABLE_ROUTER_SELECT_FAST` | kill switch | Full 512-wide bitonic sort for router top-k instead of the iterative selection (bit-identical). | `ds4_metal.m:3437` |
| `DS4_GLM_DISABLE_ROUTER_SHARED_FOLD` | kill switch | Router not folded into the head of the shared-expert gate+up grid (two dispatches instead of one). | `ds4_metal.m:42125` |
| `DS4_GLM_DISABLE_ROUTER_TAIL_FOLD` | kill switch | Top-8 router selection as its own dispatch instead of folded onto the logits matvec tail. | `ds4.c:44614` |
| `DS4_GLM_DISABLE_SCORER_HALF` | kill switch | Forces the production DSA scorer kernel regardless of any variant request. | `ds4_metal.m:20279` |
| `DS4_GLM_DISABLE_SCORER_XREDUCE` | kill switch | Kill switch for the xr* scorers. | `ds4_metal.m:20286` |
| `DS4_GLM_DISABLE_SINKHORN_PAR` | kill switch | Sinkhorn comb back behind the barrier it used to wait on (no overlap with the collapse). | `ds4_metal.m:53091` |
| `DS4_GLM_DISABLE_TOOL_RESULT_REORDER` | kill switch | Disables the GLM tool-result reorder that renders parallel tool results in the order the chat template expects. | `ds4_server.c:2732` |
| `DS4_GLM_DISABLE_TOPK_FAST` | kill switch | Disables the bounded-radix DSA top-k fast path (Tier 1; its only switch; not clamped by DS4_GLM_EXACT). | `ds4_metal.m:20438` |
| `DS4_GLM_DISABLE_TOPK_ONESHOT` | kill switch | Disables the one-shot DSA top-k selection path. | `ds4_metal.m:20133` |
| `DS4_GLM_MTP_NO_ROWSNAP` | kill switch | Disables the MTP row-boundary KDA snapshot fast path (inert on real GLM-5.3 graphs on this branch; guard at ds4.c:68578). | `ds4.c:49123` |
| `DS4_KDA_CONCURRENT_DISABLE` | kill switch | Runs the six KDA first-level prefill projections serially instead of in one concurrent dispatch level. | `ds4.c:44801` |
| `DS4_KV_EVICT_RAW` | kill switch | =1 restores the raw full-length KV store on eviction instead of trimming it to the last client-transcript position (which keeps disk keys reproducible). | `ds4_server.c:10431` |
| `DS4_METAL_DISABLE_GLM53_Q8_QKV` | kill switch | Disables the fused Q8_0 KDA q/k/v decode matvec (separate projections instead). | `ds4.c:45373` |
| `DS4_SLOT_SCORE_LEGACY` | kill switch | Restores the legacy slot placement instead of eviction-cost scoring (only matters with --batched-session >= 2). | `ds4_server.c:13682` |
| `DS4_STREAM_GUARD_LEGACY` | kill switch | Restores the old streaming guard that held all answer text until a second </think> when thinking and tools were both enabled. | `ds4_server.c:6553` |
| `DS4_DFLASH_MIN_MARGIN` | experimental opt-in | Opt-in draft-confidence floor: drops trailing drafts whose top1-top2 logit margin is below N. | `ds4_dflash2.inc:1605` |
| `DS4_GLM_ENABLE_HCX_NR` | experimental opt-in | Selects an HC-expand row-count screening variant (nr2w/nr4/nr4w); mutually exclusive with ptail; not retained as a default. | `ds4_metal.m:51823` |
| `DS4_GLM_ENABLE_HCX_VECHC` | experimental opt-in | Selects the float4 HC post/comb epilogue on M3 (gated to M5 by upstream policy) for screening. | `ds4_metal.m:51798` |
| `DS4_GLM_ENABLE_KDA_PROJ_FUSE` | experimental opt-in | All-Q8_0 fusion of the whole KDA projection stage (8 dispatches to 2 per layer); reachable only on an all-Q8 KDA layout such as the public artifact and not yet validated there. | `ds4.c:45302` |
| `DS4_GLM_ENABLE_MOE_BLOCK_DATAFLOW` | experimental opt-in | =1 runs the whole MoE block as one persistent dispatch (bit-exact, measured slower in graph). | `ds4_metal.m:42381` |
| `DS4_GLM_ENABLE_PREFILL_FOLD_MOESWIGLU_ROUTED` | experimental opt-in | =1 width-4 SwiGLU on the routed half too (measured no gain). | `ds4_metal.m:3374` |
| `DS4_GLM_ENABLE_SCORER_XREDUCE` | experimental opt-in | =1 enables the transposed-butterfly scorer xr8 above the depth gate (Tier 2, registry entry 5; opt-in for the reason in FIDELITY.md). | `ds4_metal.m:20254` |
| `DS4_GLM_ENABLE_SPLIT8_DBLBUF` | experimental opt-in | =s8d1|s8d0|s4d1|s16d0 double-buffered split8 staging (Tier 1; registry entry 10, off under DS4_GLM_EXACT). | `ds4_metal.m:40266` |
| `DS4_GLM_ENABLE_SPLIT8_VPLANE` | experimental opt-in | =1 split8 reduce value projection via simd_sum (Tier 2; registry entry 11). | `ds4_metal.m:40329` |
| `DS4_GLM_HC_PHASEC_HOIST` | experimental opt-in | =1 hoists the hc_pre phase-C comb (A/B variant of the tail kernel). | `ds4_metal.m:53092` |
| `DS4_GLM_HC_PRE_ALGEBRA_B` | experimental opt-in | =1 enables hc_pre algebra half B (sliced collapse with communicated sum of squares; bit-exact but slower). | `ds4_metal.m:53059` |
| `DS4_GLM_HC_PRE_ONE_DISPATCH` | experimental opt-in | =1 runs the hc_pre pair as one dispatch with a last-threadgroup tail (bit-identical, measured slower). | `ds4.c:45038` |
| `DS4_GLM_KDA_PAIR2IN_FLAT` | experimental opt-in | Selects the flat variant of the BF16 pair2in kernel. | `ds4_metal.m:53842` |
| `DS4_GLM_ROUTED_ASTAGE_DB` | experimental opt-in | =1 enables A-stage double buffering in the routed-expert prefill GEMM (one barrier per k-step). | `ds4_metal.m:43823` |
| `DS4_GLM_ROUTER_NR0` | experimental opt-in | =1 one-row-per-threadgroup router logits kernel (measured indistinguishable; default 2). | `ds4_metal.m:41929` |
| `DS4_GLM_SHARED_MID_NR0` | experimental opt-in | =1 one-row-per-threadgroup shared-expert mid kernel (within noise; default 2). | `ds4_metal.m:23202` |
| `DS4_MV_EXT_R1_8` | experimental opt-in | Selects the r1_8 extended matvec variant (measured slower on M3 Ultra; kept for experiments). | `ds4_metal.m:6404` |
| `DS4_DFLASH_FORCE_DRAFTS` | developer instrumentation (bench-only) | Caps a block at N drafts, so the verify block is N+1 rows including the anchor. It caps the length only; it does not choose the rejection position. | `ds4_dflash_glm.inc:264` |
| `DS4_DFLASH_GOLDEN_DIR` | developer instrumentation (bench-only) | Loads drafter goldens from a directory at engine load and exits the process afterwards. | `ds4_dflash_golden.inc:50` |
| `DS4_DFLASH_STATS` | developer instrumentation (bench-only) | Per-cycle acceptance and timing statistics, plus a totals summary at engine close. | `ds4_dflash2.inc:1638` |
| `DS4_DFLASH_STEP_PROFILE` | developer instrumentation (bench-only) | Per-stage timing inside a draft step. | `ds4_dflash2.inc:1329` |
| `DS4_DFLASH_VERIFY_ROWS` | developer instrumentation (bench-only) | Selects the alternate row-verify route (byte-identical to the batch route, not faster). | `ds4_dflash_glm.inc:357` |
| `DS4_DFLASH_ZERO_FEATURES` | developer instrumentation (bench-only) | Zeroes the drafter's input features. It makes a mismatch likely, not certain, and does not choose the rejection position: use `DS4_DFLASH_SCRIPT` for that. | `ds4_dflash_glm.inc:270` |
| `DS4_GLM_BF16_LOWRANK_SPLITK_SLICES` | developer instrumentation (bench-only) | Forces the split-K slice count. | `ds4_metal.m:52188` |
| `DS4_GLM_BF16_LOWRANK_SPLITK_TGS` | developer instrumentation (bench-only) | Threadgroups per tile for the split-K (default 16; the gated shape). | `ds4_metal.m:52129` |
| `DS4_GLM_DENSE_HALF_COPY_MIN_ROWS` | developer instrumentation (bench-only) | Row threshold for the half copy. | `ds4_metal.m:21166` |
| `DS4_GLM_DENSE_HALF_COPY_STATS` | developer instrumentation (bench-only) | Prints half-copy hit statistics. | `ds4_metal.m:21177` |
| `DS4_GLM_DSA_ATTN_FP64_REF` | developer instrumentation (bench-only) | Runs an FP64 sequential host reference of DSA attention alongside the kernel (minutes per prompt; never a shipping path). | `ds4_metal.m:41192` |
| `DS4_GLM_DSA_BATCH_PIN` | developer instrumentation (bench-only) | =1 selects the fma-rounded accumulate that is NOT bit-identical (negative control for the harness). | `ds4_metal.m:40887` |
| `DS4_GLM_DSA_BLOCKED_BLOCK` | developer instrumentation (bench-only) | Gather stage rows for the blocked softmax (sweep knob; default 24). | `ds4_metal.m:41034` |
| `DS4_GLM_DSA_BLOCKED_CTRL` | developer instrumentation (bench-only) | Control-arm selector for the blocked-softmax harness. | `ds4_metal.m:41098` |
| `DS4_GLM_DSA_BLOCKED_NOPART` | developer instrumentation (bench-only) | Disables the partial-block path (sweep knob). | `ds4_metal.m:41087` |
| `DS4_GLM_DSA_BLOCKED_NOSKIP` | developer instrumentation (bench-only) | Disables block skipping (sweep knob). | `ds4_metal.m:41086` |
| `DS4_GLM_DSA_BLOCKED_SUB` | developer instrumentation (bench-only) | Softmax sub-block rows (sweep knob; default 4). | `ds4_metal.m:41090` |
| `DS4_GLM_DSA_BLOCKED_TWOPASS` | developer instrumentation (bench-only) | Two-pass instead of one-pass blocked softmax (sweep knob). | `ds4_metal.m:41088` |
| `DS4_GLM_GEN_COUNTERS` | developer instrumentation (bench-only) | Prints the decode loop's own counters and stop reason after generation. | `ds4.c:56153` |
| `DS4_GLM_HC_CHAIN_SCALE_STANDALONE` | developer instrumentation (bench-only) | Takes the HC scale from a standalone reduction so the two halves of the SCALE fold can be measured apart. | `ds4.c:44660` |
| `DS4_GLM_HC_PRE_NSG` | developer instrumentation (bench-only) | Simdgroups per threadgroup for the hc_pre split-K kernel (selected together with the slice count). | `ds4_metal.m:52838` |
| `DS4_GLM_HC_TAIL_REPL_TGS` | developer instrumentation (bench-only) | Replica count for the wide hc_pre tail (default 13; =9 restores the earlier shape; bit-exact at every N >= 2). | `ds4_metal.m:52964` |
| `DS4_GLM_HC_TAIL_SLICE_TGS` | developer instrumentation (bench-only) | Threadgroup count for the sliced hc_pre tail (algebra half B). | `ds4_metal.m:53041` |
| `DS4_GLM_HC_TAIL_SPIN_CAP` | developer instrumentation (bench-only) | Watchdog spin cap for the sliced tail's counter wait; 0 forces the recompute path for testing. | `ds4_metal.m:53030` |
| `DS4_GLM_IDXDEC_DOUBLE` | developer instrumentation (bench-only) | Repeats the split8 partial/reduce dispatches N times for timing. | `ds4_metal.m:40413` |
| `DS4_GLM_IGNORE_EOS` | developer instrumentation (bench-only) | Keeps decoding to -n past EOS so paired benchmarks get equal decode windows; changes the generated text. | `ds4.c:56101` |
| `DS4_GLM_INDEXER_HEADFOLD` | developer instrumentation (bench-only) | Selects a head-fold variant by name (g2e2 .. g8e8a). | `ds4_metal.m:710` |
| `DS4_GLM_INDEXER_STREAM_THREADS` | developer instrumentation (bench-only) | Threadgroup width of the streamed scorer. | `ds4_metal.m:20356` |
| `DS4_GLM_KDA_PREFILL_BLOCK` | developer instrumentation (bench-only) | Tokens per KDA prepare threadgroup (sweep knob). | `ds4_metal.m:55543` |
| `DS4_GLM_KDA_PREFILL_COLS` | developer instrumentation (bench-only) | Value columns per simdgroup in the KDA recurrence (sweep knob). | `ds4_metal.m:55565` |
| `DS4_GLM_KDA_PREFILL_PREFETCH` | developer instrumentation (bench-only) | Prefetch mode of the recurrence (sweep knob). | `ds4_metal.m:55657` |
| `DS4_GLM_KDA_PREFILL_RTHREADS` | developer instrumentation (bench-only) | Threadgroup width of the plain/wide recurrence (sweep knob). | `ds4_metal.m:55591` |
| `DS4_GLM_KDA_PREFILL_STAGED` | developer instrumentation (bench-only) | Stage q/k/decay through threadgroup memory (sweep knob). | `ds4_metal.m:55647` |
| `DS4_GLM_KDA_PREFILL_THREADS` | developer instrumentation (bench-only) | Threadgroup width of the staged recurrence (sweep knob). | `ds4_metal.m:55635` |
| `DS4_GLM_KDA_PREFILL_VEC` | developer instrumentation (bench-only) | float4 moves in the recurrence (sweep knob). | `ds4_metal.m:55579` |
| `DS4_GLM_LOAD_PAYLOAD` | developer instrumentation (bench-only) | Bench-only: restores a prepared prefix from a file instead of prefilling the prompt. | `ds4_cli.c:585` |
| `DS4_GLM_MOE_BLOCK_GRID` | developer instrumentation (bench-only) | Resident threadgroups for the MoE block kernel (default 240). | `ds4_metal.m:42398` |
| `DS4_GLM_MOE_BLOCK_NOBATCH` | developer instrumentation (bench-only) | MoE block kernel without batched segments (diagnostic). | `ds4_metal.m:42674` |
| `DS4_GLM_MOE_BLOCK_NOINTERLEAVE` | developer instrumentation (bench-only) | MoE block kernel without expert-interleaved segments (diagnostic). | `ds4_metal.m:42682` |
| `DS4_GLM_MOE_BLOCK_ONLY` | developer instrumentation (bench-only) | Restricts the MoE block kernel to one phase (diagnostic). | `ds4_metal.m:42684` |
| `DS4_GLM_MOE_BLOCK_STUB` | developer instrumentation (bench-only) | Compiles the dependency waits out (numerically wrong; prices the counters). | `ds4_metal.m:42673` |
| `DS4_GLM_MOE_BLOCK_VERBOSE` | developer instrumentation (bench-only) | Prints MoE block kernel diagnostics for early layers/positions. | `ds4.c:46519` |
| `DS4_GLM_PREFILL_TRACE` | developer instrumentation (bench-only) | =1 turns on per-stage host timers inside the prefill chunk. | `ds4.c:38137` |
| `DS4_GLM_PREFILL_TRACE_ALL` | developer instrumentation (bench-only) | Prints every prefill stage, not just slow ones. | `ds4.c:38142` |
| `DS4_GLM_PREFILL_TRACE_SLOW_MS` | developer instrumentation (bench-only) | Threshold in ms for a prefill stage to be printed. | `ds4.c:38147` |
| `DS4_GLM_ROUTED_DEQ_WIDE` | developer instrumentation (bench-only) | =4|16 selects the dequant load width for A/B. | `ds4_metal.m:43836` |
| `DS4_GLM_ROUTED_GATEUP_WIDE` | developer instrumentation (bench-only) | Selects the routed gate+up load-width variant 0..4 for A/B (only bit-exact forms reachable). | `ds4_metal.m:42935` |
| `DS4_GLM_ROUTED_GROUPED_MIN_TOKENS` | developer instrumentation (bench-only) | Token count above which the grouped routed path replaces the decode-style expert matvecs in prefill. | `ds4_metal.m:44014` |
| `DS4_GLM_ROUTED_TILE_MODE` | developer instrumentation (bench-only) | Forces a routed prefill tile mode from the internal table. | `ds4_metal.m:43855` |
| `DS4_GLM_ROUTED_TILE_NR1` | developer instrumentation (bench-only) | Forces the routed prefill tile's nr1 for sweeps. | `ds4_metal.m:43856` |
| `DS4_GLM_ROUTER_SHARED_FOLD_VERBOSE` | developer instrumentation (bench-only) | Prints why the router/shared-expert fold was refused. | `ds4_metal.m:42157` |
| `DS4_GLM_ROUTER_TILE_NROW` | developer instrumentation (bench-only) | Rows per threadgroup in the router prefill kernel (A/B). | `ds4_metal.m:24890` |
| `DS4_GLM_ROUTER_TILE_NTOK` | developer instrumentation (bench-only) | Tokens per tile in the router prefill kernel (A/B). | `ds4_metal.m:24888` |
| `DS4_GLM_SCORER_CAPTURE` | developer instrumentation (bench-only) | =<prefix> writes the DSA indexer scorer operands to files. | `ds4.c:17349` |
| `DS4_GLM_SCORER_CAPTURE_STEPS` | developer instrumentation (bench-only) | Limits the scorer capture to N steps. | `ds4.c:17353` |
| `DS4_GLM_SCORER_VARIANT` | developer instrumentation (bench-only) | Forces a DSA scorer variant by name (Tier 2 names are still clamped by the kill switch and DS4_GLM_EXACT). | `ds4_metal.m:20267` |
| `DS4_GLM_SCORER_XREDUCE_SHAPE` | developer instrumentation (bench-only) | Selects the xr* shape when enabled. | `ds4_metal.m:20256` |
| `DS4_GLM_TOPK_CAPTURE` | developer instrumentation (bench-only) | =<file> writes the top-k selector's input scores per (step, layer) for the offline certification harness. | `ds4.c:17276` |
| `DS4_GLM_TOPK_FAST_BITS` | developer instrumentation (bench-only) | Radix key bits for the fast path (harness sweep). | `ds4_metal.m:20484` |
| `DS4_GLM_TOPK_FAST_STATS` | developer instrumentation (bench-only) | Prints calls / accepted calls of the fast path at exit. | `ds4_metal.m:20535` |
| `DS4_GLM_TOPK_FAST_TGS` | developer instrumentation (bench-only) | Threadgroups for the fast path's full-row passes (harness sweep). | `ds4_metal.m:20500` |
| `DS4_GLM_TOPK_FUSED_GROUP` | developer instrumentation (bench-only) | Fused merge levels in the top-k merge chain. | `ds4_metal.m:20148` |
| `DS4_GLM_TRACE_SELECTED` | developer instrumentation (bench-only) | Reports at each indexed dispatch whether tail sentinels remain in the selection buffer. | `ds4.c:17123` |
| `DS4_KERNEL_LEDGER` | developer instrumentation (bench-only) | Per-kernel decode bandwidth ledger (mode number). | `ds4_metal.m:1770` |
| `DS4_KERNEL_LEDGER_BY_TG` | developer instrumentation (bench-only) | Ledger broken down by threadgroup count. | `ds4_metal.m:1775` |
| `DS4_KERNEL_LEDGER_DUMP` | developer instrumentation (bench-only) | =<path> dumps the ledger to a file. | `ds4_metal.m:1880` |
| `DS4_KERNEL_LEDGER_RESET_AFTER` | developer instrumentation (bench-only) | Resets the ledger after N evaluations (skips warm-up). | `ds4_metal.m:1776` |
| `DS4_METAL_DFLASH2_SOURCE` | developer instrumentation (bench-only) | Overrides the path of metal/dflash2.metal when the Metal library is assembled (upstream's per-file override mechanism). | `ds4_metal.m:5562` |
| `DS4_METAL_GLM53_MOE_BLOCK_SOURCE` | developer instrumentation (bench-only) | Overrides the path of metal/glm53_moe_block.metal. | `ds4_metal.m:5569` |
| `DS4_METAL_PRETOUCH_TENSORS` | developer instrumentation (bench-only) | Faults every owned tensor's pages in on the CPU at allocation so a measurement can separate buffer size from first-touch cost. | `ds4_metal.m:4672` |
| `DS4_METAL_T2SCREEN_SOURCE` | developer instrumentation (bench-only) | Overrides the path of metal/t2screen.metal (which holds the default ptail HC-expand kernel and the scorer variants). | `ds4_metal.m:5584` |

Upstream's own switches (`DS4_METAL_Q8_MV_NSG`, `DS4_TP_*`, `DS4_DSPARK_*`,
`DS4_METAL_ENCODER_TIMELINE`, …) are all present and unchanged.

## Benchmark recipes

The instruments and their environment are described in
[`bench/README.md`](../bench/README.md). The public-model capability commands use the
model digest in "Tested configuration" and the exact prompt digests listed in
[`bench/RELEASE-EVIDENCE.md`](../bench/RELEASE-EVIDENCE.md). Set the prompt variables to
local files with those digests before running:

```sh
export MODEL=gguf/GLM-5.3-Flash-Q4_K.gguf
export PROMPT62=/path/to/needle-64k.txt
export PROMPT300=/path/to/prompt-300k.txt
export DS4_GLM53_MEMORY_CEILING_GB=280
export DS4_GLM_GEN_COUNTERS=1 DS4_GLM_IGNORE_EOS=1

# Serial native generation gate. The final guarded-profile build resolves model
# mapping and expert-bank defaults itself; force flags below reproduce b723dfa only.
./ds4 -m "$MODEL" --metal --dflash-mode serial --nothink --temp 0 \
      -c 70000 -n 2048 --prompt-file "$PROMPT62"
./ds4 -m "$MODEL" --metal --dflash-mode serial --nothink --temp 0 \
      -c 320000 -n 2048 --prompt-file "$PROMPT300"
```

A native decode block counts only when the normal CLI session loop reports
`generated=2048`, `requested=2048`, `stop=limit`, and 2,047 committed forward
positions after the initial sampled token. One valid complete block records achieved
capability. Interleaved matched runs are still required to attribute a speed difference
or characterize repeatability. `DS4_GLM_IGNORE_EOS=1` is a benchmark-only fixed-horizon
control and changes the generated text.

The `b723dfa` capability receipts forced the bank, fused layer command buffer,
eight-layer pipeline, `xr8` at its depth gate, untracked model view, and serial DFlash.
They are evidence for that named configuration. The final release recipe should use the
guarded defaults after their implementation and record the resolved startup policy.

For an exact `b723dfa` configuration reproduction, export
`DS4_METAL_MODEL_UNTRACKED=1`, `DS4_GLM_ENABLE_EXPERT_BANK=1`,
`DS4_GLM_EXPERT_BANK_FUSED_LAYER_CB=1`,
`DS4_GLM_EXPERT_BANK_PIPELINED_LAYERS=1`,
`DS4_GLM_ENABLE_SCORER_XREDUCE=1`, and
`DS4_GLM_DISABLE_ROUTER_SPLITK_B4=1` before the native commands.

**Quality screen:**

```sh
make -C gguf-tools quality-score
DS4_GLM53_MEMORY_CEILING_GB=400 \
  gguf-tools/quality-testing/score_official "$MODEL" \
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv \
  fast.tsv
DS4_GLM_EXACT=1 DS4_GLM53_MEMORY_CEILING_GB=400 \
  gguf-tools/quality-testing/score_official "$MODEL" \
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv \
  exact.tsv
python3 gguf-tools/quality-testing/compare_scores.py upstream-9ab7053.tsv exact.tsv
python3 gguf-tools/quality-testing/compare_scores.py upstream-9ab7053.tsv fast.tsv
```

Compare scores and outputs as measured data. Exact mode disables the registered Tier 2
changes, but the corrected DSA pad-row semantics remain active, so upstream TSV byte
identity is not assumed.

## Results

These are capability receipts from `b723dfa7594dbadc6480cf2872751ae22e551754`
(`ds4` sha256 `31139ed7…`) on the public 185,299,232,064-byte GGUF. They are not yet
receipts for the final rebuilt artifact.

| measurement | `b723dfa` result | scope | staging receipt |
|---|---:|---|---|
| native prefill, 62,174 prompt tokens | **550.72 t/s** | forced shared bank + fused layer command buffer + eight-layer pipeline; all 42 layers banked | `ASTRAL-BANK-PIPELINED62-20260906T194550Z` |
| native serial decode, 300,000-token prompt | **37.2993996 t/s** | 2,048 generated / 2,047 evaluated, stop at prediction limit | `BASE300-astral-b723-pipeline-20260906T195448Z` |
| native prefill, 300,000 prompt tokens | **473.75 t/s** | five admitted bank groups, no refusal | same |
| DFlash SQL fixed horizon, serial / conservative / speculative | **38.5850 / 47.2064 / 60.7875 t/s** | 8,192 generated in every arm; favorable repetitive prompt, truncated during tuple 483 of 2,000 | `ASTRAL-THREE-MODES-20260906T192241Z` |
| task outcome screen | **77/77 in both arms** | 22/23 outputs byte-identical; one wording difference; focused screen, not broad equivalence | `TASKCHECK-20260906T154355Z` |
| E6 token-weighted NLL margin | **unmet** | candidate 1.270792 vs upstream 1.268859, delta +0.001933; provisional +0.0005 criterion not met | E6 retained comparison |

The 300k output matched the retained v3+`xr8` reference block. That is lineage evidence,
not a claim of universal or upstream byte identity. The DFlash arms happened to emit the
same 22,036 bytes on the SQL fixture. They reached the fixed token horizon rather than
completing the requested task, so the result must not be presented as average workload
speed.

The concise evidence ledger and final-release placeholders are in
[`bench/RELEASE-EVIDENCE.md`](../bench/RELEASE-EVIDENCE.md). Binary, source, model and
prompt identities belong in each retained receipt; the final release rows stay pending
until the rebuilt artifact is screened.

## Known issues

- **`/v1/models` listed only GLM-5.2 alias ids for a GLM-5.3 model — fixed.** Upstream's
  listing keyed the DSA family by its first member. `send_models()` now leads with the
  loaded model's id and keeps the GLM-5.2 ids as documented legacy aliases (see "Model
  ids" above); lifecycle test T13 checks the listing. Receipts from before this fix
  (`RUN-20260906T12…`) show the old listing.
- **Negative `max_tokens` was accepted and clamped to 0 — fixed.** `max_tokens`,
  `max_completion_tokens` and `max_output_tokens` must now be a non-negative integer
  (`json_max_tokens()`, all four request parsers); anything else is a 400 in the server's
  usual error shape, `{"error":{"message":"invalid max_tokens: must be a non-negative
  integer","type":"invalid_request_error"}}`. Zero is still accepted and yields an empty
  completion with `finish_reason: length`. Lifecycle test T14 checks it; T7b's
  observational miss in the earlier receipts is this.
- **DFlash2 is optional.** Bare startup is serial. With local draft weights,
  conservative request-credit scheduling is the default; the uncapped speculative mode
  is explicit and can regress. The less-than-2% figure is the conservative scheduler's
  empirical added-work target on its calibrated profile, not a promise for every
  workload or machine. Positive-temperature requests use the serial path.
- **Repeat of a long prompt accounted as a cache hit while the whole prompt was
  rebuilt — fixed on two paths.** Two mechanisms are involved. First, upstream's
  validity check on the memory-rewind path (`ds4_server.c`, the `rewind_valid` test after
  `ds4_session_rewind`: the servable common prefix must equal the rewind target, and the
  vision state must match) correctly rejects an invalid live rewind — after a GLM-5.3
  live-prefix rewind that cannot restore the recurrent state the session reports a
  servable prefix of 0 and the log says `GLM live prefix rewind from N to M requires
  rebuild`. Second, the text-key path: `live_text_prefix_prompt()` (`ds4_server.c:10678`)
  then matched the rendered text of the live tokens and returned the live token count
  without checking that the session could serve it, so the request was accounted as
  cached (`cache_source memory-text`, `cached_tokens 7014`) while the whole prompt was
  rebuilt, and the disk-snapshot path — tried only when nothing live is claimed — was
  skipped. Observed on the public build `a0bf48d` as lifecycle test T9 (receipt
  `server-lifecycle/RUN-20260906T121302Z`): a 7,015-token
  prompt repeated, 15.67 s then 15.33 s, no saved work, although a 6,144-token cold
  snapshot had been stored seconds earlier. This fork's fix, commit `8d7e091` ("server:
  validate the live text-prefix hit against the session before claiming it"), checks the
  servable prefix under `inference_mu` and declines the live claim when the session
  cannot serve the whole live prefix (`live text prefix of N tokens matched but only M
  are servable; not claiming it (disk snapshot may be used)`), so the disk snapshot is
  used. Receipt on the v2 candidate (`public-glm53` + `8d7e091` + the KDA test commits;
  `ds4-server` sha256 `8bc890e9…`, ctx 65536):
  `server-lifecycle/RUN-20260906T123859Z` — the repeat
  of the 7,015-token prompt logged the rewind rejection, the declined text claim, then
  `kv cache hit text tokens=6144 … load=36.9 ms` and `chat ctx=6144..7015:871` prefill
  over exactly the 871-token suffix (`server.log` 138–147); accounting `cache_source
  disk-text`, `cached_tokens 6144`; wall time 13.76 s → 1.93 s (`results.json` T9a/T9b).
  Scope of that evidence: the same-process disk fallback in this sequence (cold prefill,
  one cold snapshot stored, immediate repeat). Restart, session reset, eviction and other
  state formats are not covered by it and remain open; vision re-validation is still
  pending.
- **`--help glm53`** (all five tools) lists the supported controls and kill switches,
  and `--help runtime` lists `--dflash`; the experimental opt-ins and the developer
  instrumentation are documented only here.
- **Supported kernels live in a file named for the screening campaign.** The default
  `ptail` HC-expand kernel and the DSA scorer variants are defined in
  `metal/t2screen.metal`; relocating the supported kernels into a production-named
  Metal file with a build/runtime check is future work, listed below.

## Checklist for publication

- [ ] Build the final merged artifact and replace the pending column in
      `bench/RELEASE-EVIDENCE.md` with its binary/source identities and retained
      repository-relative receipts.
- [ ] Confirm the guarded M3 Ultra/public-Q4 startup policy, including the untracked
      model view, expert bank, fused layer command buffer and eight-layer pipeline.
- [ ] Run the focused 32k auto-vs-disabled screen before freezing the provisional
      32768-token expert-bank boundary.
- [ ] Record the exact-mode upstream diagnostic on the final artifact; report measured
      deltas and do not assume byte identity because the DSA pad-row fix remains active.
- [ ] Re-validate a vision prompt after the multimodal session-reuse change.
- [ ] Re-run only the affected DFlash startup/lifecycle checks on the final artifact;
      keep the `b723dfa` 16/16 result as capability evidence rather than repeating
      unchanged broad matrices.
- [ ] Decide DFlash2 sampled operation: compute the rejection residual against the
      target's original filtered support instead of masking a logit, or leave the mode
      greedy-only.
- [ ] Decide the MTP row-snapshot path: extend the snapshot to the indexer tail, or
      remove the path and `DS4_GLM_MTP_NO_ROWSNAP`.
- [ ] Relocate the supported kernels out of `metal/t2screen.metal` into a
      production-named Metal file, with a build/runtime check that the default dispatch
      still resolves; then decide the screening-only variants (`HCX_NR`, `HCX_VECHC`).
- [ ] Per row of the developer-instrumentation class: keep, gate behind a build flag, or
      delete (`DS4_GLM_LOAD_PAYLOAD` and the `DS4_DFLASH_GOLDEN_DIR` exit-after-load path
      first).
- [ ] Screen `DS4_GLM_ENABLE_KDA_PROJ_FUSE` on the public artifact (identity vs the
      default required for any default-on).
