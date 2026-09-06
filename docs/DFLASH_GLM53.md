# DFlash2 speculative decoding for GLM-5.3-Flash

`--dflash` is an optional speculative-decoding mode. A small block-diffusion
drafter proposes a block of tokens, the target model verifies all of them in
one forward, and the longest prefix the target agrees with is committed. When
speculation pays, decode is faster; when it does not, the mode measures that
and parks itself on plain serial decode.

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

**Status of this page.** The conversion and provenance steps below are
verified: the pinned source hash was computed here, and re-running the pinned
converter command reproduced the tested artifact byte for byte. The `b723dfa`
capability build also passed the affected startup, server reuse, cancellation,
stop and natural-EOS runtime checks. The fixed-horizon measurement in §1 shows
that this drafter can pay on one favorable structured workload. It does not
establish a general speedup. DFlash2 remains optional, greedy-only and dependent
on a locally obtained drafter. The final rebuilt release artifact still needs
its own retained runtime receipt.

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

### Capability receipt on `b723dfa`

One deliberately repetitive SQL prompt was run for a fixed 8,192-token horizon
through all three startup policies. Each arm emitted 22,036 bytes with the same
digest in that run:

| startup policy | generated tokens | generation time | tokens/s |
| --- | ---: | ---: | ---: |
| serial | 8,192 | 212.310 s | 38.5850 |
| conservative | 8,192 | 173.536 s | 47.2064 |
| speculative | 8,192 | 134.764 s | 60.7875 |

Receipt: `ASTRAL-THREE-MODES-20260906T192241Z`, build `b723dfa`. The prompt asks
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
      --ctx 65536 -p "your prompt"
```

Server: the same `--dflash FILE` flag.

```sh
./ds4-server --model /path/to/GLM-5.3-Flash-Q4_K-9ab7053.gguf \
             --dflash "$DFLASH_DIR/GLM-5.3-Flash-DFlash2.gguf" \
             --dflash-mode conservative \
             --ctx 65536
```

`--dflash-mode` is a startup policy shared by the CLI and server:

- With neither `--dflash` nor an explicit mode, startup is serial and allocates
  no drafter state.
- `conservative` uses request-credit admission. This is also the default when
  `--dflash FILE` is supplied without an explicit mode.
- `speculative` runs the existing uncapped path. It can improve or regress
  throughput depending on the prompt, context, and acceptance rate.
- `serial` does not load the draft model or allocate DFlash scratch, even when
  a `--dflash FILE` path is present.

Speculative and conservative modes require `--dflash FILE`; without draft
weights they fail at startup. All three modes retain the target model as the
verifier and are intended to produce the same quality. They differ in whether
and when speculative work is scheduled. The conservative policy targets less
than 2% added generated-work time on its calibrated M3 Ultra profile; that is
an empirical admission target, not a universal hardware guarantee.

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
| `DS4_DFLASH_NO_ADAPTIVE=1` | With no explicit `--dflash-mode`, select the uncapped speculative policy. |
| `DS4_DFLASH_STATS=1` | Per-cycle drafted/accepted counts, rollback kind and stage timings on stderr, plus a totals summary when the engine closes. |
| `DS4_DFLASH_ZERO_FEATURES=1` | Zero the drafter's input features. It makes a mismatch *likely*, not certain: the drafter still emits a deterministic token per row, and a common one occasionally coincides with the target's own prediction. |
| `DS4_DFLASH_FORCE_DRAFTS=N` | Cap every block at N drafted tokens, so the verify block is N+1 rows including the anchor. It caps the block length; it does **not** choose where a rejection lands. |
| `DS4_DFLASH_CTX_CAP=N` | Cap the drafter's context rows (default ~256). |
| `DS4_DFLASH_SCRIPT=P:K:N,...` | Deterministic draft supply (test-only): at absolute position P propose N drafts of which the first K are the retained serial continuation, with a known-wrong token at row K. The verifier is untouched and decides for itself. Unnamed positions decode serially, so blocks land exactly where asked. See `tests/dflash_rejection_harness.sh`. |
| `DS4_DFLASH_SCRIPT_IDS=FILE` | The retained continuation the script proposes from: a base position followed by one token id per line. |
| `DS4_DFLASH_SCRIPT_SERIAL=1` | Control arm: same binary, same dumps, every position serial. |
| `DS4_DFLASH_SCRIPT_DUMP=DIR` | At each scripted frontier, write the frontier logits and a per-tensor digest of the complete speculative state. |
| `DS4_DFLASH_FAIL=point[,...]` | Inject a cycle failure at `state_save`, `after_arm` or `after_verify`, to exercise the cleanup on those exits. |

The full table of DFlash switches is in `docs/GLM53_M3ULTRA.md`.

## 6. How rejection is undone

A verify forward advances more per-token state than the tokens it commits. On
GLM-5.3 that state is the KDA convolution and recurrent buffers **and**, on
graphs with full DSA layers, the indexer tail K+gate ring. If part of a
drafted block is rejected, all of that has to go back to the frontier the
committed prefix reached — otherwise the rejected rows stay in the ring while
the checkpoint advances, and the next token is computed against state that no
serial run would ever have produced.

This branch handles it the same way the MTP speculative cycle handles its own
rejections: before the verify it saves the **complete** speculative state, and
on a partial acceptance it restores that state and replays the committed
prefix one token at a time through the ordinary serial forward. Every
per-token producer is re-derived rather than patched, and the frontier logits
are the serial path's own. Where the graph has no indexer tail, the cheaper
per-step snapshot taken inside the verify is still used, because there it
covers the whole state.

The replay costs real time — a rejection pays for the tokens it keeps twice.
That is visible to the throttle (section 7) as cycle wall time, so blocks that
reject often push the mode towards parking on their own.

## 7. Adaptive behaviour, and what it does and does not promise

The throttle measures two things live: the wall time of a speculative cycle
and the wall time of a serial token. Their ratio is the break-even acceptance
— the number of tokens a cycle must commit to be worth running. An EMA of
recent commit counts is compared against it. Below break-even the mode parks
on serial decode and buys an occasional paired probe (one cycle to refresh the
drafter's features, one to measure honestly) on an exponentially backed-off
cadence; a probe that beats break-even re-engages immediately.

**Greedy only.** `--dflash` speculates for greedy decoding. A request with a
positive temperature decodes serially instead, and this is deliberate: the
sampled cycle signals a rejection by masking the rejected token's logit, and
every caller then re-applies top-k/top-p/min-p to that modified vector — which
filters the residual against a different support than the target's own
distribution and can admit a token the original filter excluded.
`tests/test_dflash_sampling.c` demonstrates it on the real sampler. Fixing it
means computing the residual against the original filtered support; until
that exists, positive temperature does not enter the cycle.

This is a measured heuristic, not a guarantee. Specifically:

- It prices what it can time. Some setup sits outside the cycle timer: the
  tap-feature capture and readback during **prefill**, and the first cycle's
  ingest of that feature ring into the drafter's context cache. With
  `DS4_DFLASH_STATS=1` the engine prints those separately at close, under
  "outside the cycle timer", so a matched run can report them rather than
  assume them.
- One cost has no separable timer at all: the tap capture inside the prefill
  kernels themselves. Measuring it needs a matched prefill with and without
  `--dflash`.
- Model load time (opening and binding a 2.2 GiB drafter) is startup cost and
  is not part of any decode figure.
- No fixed overhead percentage is claimed. Whether the mode is worth enabling
  on a given workload is an empirical question; run it both ways.

Speculation helps some prompts and not others, and its benefit shrinks as
context grows. It is optional for that reason.

## 8. Smoke test

Confirm the mode is really running rather than quietly falling back:

```sh
DS4_DFLASH_STATS=1 ./ds4 --model TARGET.gguf --dflash "$DFLASH_DIR/GLM-5.3-Flash-DFlash2.gguf" \
    --dflash-mode speculative --ctx 8192 -n 64 \
    -p "Write a short paragraph about the sea."
```

Expect per-cycle lines with **nonzero** `drafted=` and `accepted=`, and a
closing summary:

```
ds4: dflash cycle drafted=7 accepted=4 rollback=replay ctx=... cyc=...ms be=...
ds4: dflash totals: cycles=N drafted=N accepted=N committed=N (accept_rate=...) rollback stepsnap=0 replay=N replay_tokens=N
ds4: dflash time: cycle=...ms park=...ms(N) refresh=...ms(N) | outside the cycle timer: seed_readback=...ms(N chunks, N rows) ingest=...ms(N) = ...ms
```

`drafted=0` on every line, or `rollback=refuse`, means speculation is not
actually running.

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

For this deterministic rejection harness, expect output identical to serial;
`accepted` equal to each block's K;
`rollback=replay` on every partial block and `rollback=none` on the
full-accept one; and the restored speculative state bit-identical to the
serial arm's at the same frontier (both arms reach those positions through
the same serial kernels, so the digests must match exactly — a difference is a
layout or ordering defect, not drift).

`DS4_DFLASH_NO_WIDE_ROLLBACK=1` reverts to serial decode with a one-line
notice; that path is unchanged from before and is the fallback if anything
here misbehaves.
