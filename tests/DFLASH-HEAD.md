# DFlash vocabulary-head and FC fast paths

NT4 and FC MM default on for the measured Apple M3 Ultra with at least 500 GiB
RAM and public GLM-5.3/DFlash model geometry. The applicability check includes
the 185299232064-byte target and 2342595168-byte drafter, 4096 hidden width,
154880 vocabulary, Q8 target head, and nonclassic five-layer, five-feature
drafter with block 8, window 2048, BF16 FC 20480-to-4096, FFN 12288, 32 query
heads and 8 KV heads of width 128. File sizes are applicability guards, not
content identity; no model hashing is performed. The two paths are independent
of admission: speculative mode uses them directly and conservative mode uses
them only after earning request credit. Serial mode does not use the drafter.

`DS4_DFLASH_DISABLE_HEAD_NT4=1` forces the scalar head and suppresses NT4
comparison/benchmark dispatch. `DS4_DFLASH_DISABLE_FC_MM=1` retains the original
FC dispatch. These kills dominate the legacy enable flags. The listed enable,
disable and comparison switches use presence semantics: even an empty value or
`=0` enables the switch. Unset kills
for the automatic defaults; set kills for the reference arm. The exception is
the existing `DS4_GLM_EXACT` umbrella: only a nonempty value other than `0`
disables FC MM; NT4 preserves reference arithmetic and remains eligible.

`DS4_DFLASH_HEAD_NT4=1` and `DS4_DFLASH_FC_MM=1` remain engineering forces
outside the default device/model profile. They cannot bypass a kill, the FC
exact umbrella, or the corresponding tensor shape checks. NT4 requires the
4096-to-154880 Q8 vocabulary head and eight verifier rows. Other head shapes
and failed dispatch setup retain the scalar head.

Each tile shares quantized weights and scales across four independent token
accumulators. The K assignment, eight scalar products before scaling, NSG8
reduction, and two `simd_sum` stages follow the scalar head. Normalized inputs
stay F32 and Q8 scales have the same half-to-F32 promotion as the reference.
The generic multirow matrix path has a different reduction and is not used here.
HC collapse and RMS normalization are unchanged.

`DS4_DFLASH_HEAD_COMPARE=1` computes both heads from the same resident,
normalized real verifier rows. It reports every row's full-logit word
differences, nonfinite count, maximum absolute error, both argmax IDs, and
top-two margins. It always retains scalar logits for acceptance, including
when NT4 is also requested. `DS4_DFLASH_STATS=1` records actual NT4 engagement
and whether scalar logits were retained without needing any enable flag.
These diagnostics must not be used as throughput runs. Equality on these rows
is scoped evidence, not a universal
model-quality requirement; differences require numerical and decision analysis.

`DS4_DFLASH_HEAD_BENCH=8` also enables comparison. On the first eligible block
it runs three alternating pairs of eight resident-input head repetitions.
The reference uses the existing single-dispatch exact-row scalar kernel;
NT4 uses the new shared-load tile. Reported wall milliseconds include command
encoding and completion, exclude model forward, and use a private output.
This measures the head segment, not generation throughput.

The GPU owner can add the following environment settings to the established
short JSON command without changing its model, prompt, or generation options:

```
DS4_DFLASH_NO_ADAPTIVE=1 DS4_DFLASH_STATS=1 DS4_DFLASH_HEAD_COMPARE=1 DS4_DFLASH_HEAD_BENCH=8
```

Require rows=8, nt4=1, scalar_logits_retained=1 and inspect all comparison
rows. Then price generation under explicitly uncapped speculative mode
(`--dflash-mode speculative` in the integrated CLI, or the legacy
`DS4_DFLASH_NO_ADAPTIVE=1`). Compare automatic defaults against
`DS4_DFLASH_DISABLE_HEAD_NT4=1`, holding the FC kill setting fixed. For an FC
pair, toggle only `DS4_DFLASH_DISABLE_FC_MM=1` and keep head policy fixed.
Use `DS4_DFLASH_STATS=1` to confirm actual engagement. The request-credit default
may stay serial on this short request and has a separate performance claim.
Partial final blocks remain scalar. Never leave
comparison or benchmark flags enabled in a throughput pair.

Frozen pre-experiment source is commit `2ce5473`. Its executable and runtime
Metal directory are preserved together at
`/Users/mark/megakernel-refs/public-artifact/ASTRAL-DFLASH-BASELINE-2ce5473`.
Run it from that directory to use its frozen shaders. Its executable SHA256 is
`1eedbd95bfa65d336f7440468ee0a81992943f7995d4a3626514008cdf915b5c`.

## Drafter FC arithmetic and fallback

The FC fast path sends the BF16 20480-to-4096 feature-concatenation projection
through the existing matrix path for 2..8 target rows, before its CPU output
normalization. Its temporary MV threshold is restored to the prior value on
success or failure, and a failed attempt retries the original matvec helper.
The later drafter-layer scope also restores its prior threshold, including
the cache-room failure exit. Other FC shapes retain the existing dispatch.
With stats enabled, `dflash fc rows=N mm=1` records a successful MM request,
including automatic engagement; the backend selects its normal or existing
split-K matrix implementation.

The original MV expands BF16 weights to F32, consumes F32 inputs and uses F32
FMA/reduction. MM stages both operands to FP16 and accumulates tile products in
F32. For the ordinary shipped FC shape, the existing automatic split-K rule
selects one slice. Matrix arithmetic may change draft IDs, acceptance, and
batching boundaries. The target verifier still checks all proposals and the
accepted-prefix restoration contract is unchanged. This does not promise
universal identical serial tokens: target batching already has floating-point
decision differences. No GPU-resident FC/output-normalization rewrite is included.

The frozen `f245d63` screen compared 26,019,840 full-logit words across 168 real
verifier rows and found NT4/reference identity. The separate short JSON speed
screen measured serial 39.86, old DFlash 47.54, NT4 48.35, and NT4+FC 49.26 t/s,
with the same 1432-byte output and acceptance counts. Receipts are in
`/Users/mark/megakernel-refs/public-artifact/ASTRAL-DFLASH-HEAD-SPEED-20260906T182159Z`.
These are scoped observations; 50 t/s has not been demonstrated by that screen.

`make dflash-fastpath-test` checks production force/kill/exact precedence on the
CPU, including zero-valued presence switches and unsupported shapes.
