# DFlash NT4 vocabulary-head experiment

These switches use presence semantics: even `=0` enables them. Remove a switch
with `env -u NAME` or `unset NAME` for the off arm; do not assign zero.

`DS4_DFLASH_HEAD_NT4=1` selects a four-token Q8 vocabulary-head tile only for
the shipped 4096-to-154880, eight-row verifier shape. Other shapes and a
failed dispatch setup retain the scalar head. The default is unchanged.

Each tile shares quantized weights and scales across four independent token
accumulators. The K assignment, eight scalar products before scaling, NSG8
reduction, and two `simd_sum` stages follow the scalar head. This preserves
the intended reduction structure; compiler arithmetic and actual decisions
still need measurement. The generic multirow matrix path has a different
reduction and is not used here. HC collapse and RMS normalization are unchanged.

`DS4_DFLASH_HEAD_COMPARE=1` computes both heads from the same resident,
normalized real verifier rows. It reports every row's full-logit word
differences, nonfinite count, maximum absolute error, both argmax IDs, and
top-two margins. It always retains scalar logits for acceptance, including
when NT4 is also requested. `DS4_DFLASH_STATS=1` records actual NT4 engagement
and whether scalar logits were retained. These diagnostics must not be used
as throughput runs. Equality on these rows is scoped evidence, not a universal
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
rows. Then price generation under the explicitly uncapped experiment with
`DS4_DFLASH_NO_ADAPTIVE=1 DS4_DFLASH_STATS=1 DS4_DFLASH_HEAD_NT4=1` versus the same executable with
all three head flags unset and NO_ADAPTIVE retained. The request-credit default
may stay serial on this short request and has a separate performance claim.
Partial final blocks remain scalar. Never leave
comparison or benchmark flags enabled in a throughput pair.

Frozen pre-experiment source is commit `2ce5473`. Its executable and runtime
Metal directory are preserved together at
`/Users/mark/megakernel-refs/public-artifact/ASTRAL-DFLASH-BASELINE-2ce5473`.
Run it from that directory to use its frozen shaders. Its executable SHA256 is
`1eedbd95bfa65d336f7440468ee0a81992943f7995d4a3626514008cdf915b5c`.

## Separate drafter FC experiment

`DS4_DFLASH_FC_MM=1` sends the BF16 feature-concatenation projection through
the existing matrix path for 2..8 target rows, before its existing CPU output
normalization. Its temporary MV threshold is restored to the prior value on
success or failure, and a failed attempt retries the original matvec helper.
The later drafter-layer scope also restores its prior threshold, including
the cache-room failure exit. Other FC shapes and the default remain unchanged.
With stats enabled, `dflash fc rows=N mm=1` records a successful MM request;
the backend selects its normal or existing split-K matrix implementation.

Screen this flag separately from NT4: compare acceptance, output quality and
ordinary end-to-end speed with only `DS4_DFLASH_FC_MM=1` added to the existing
JSON command. Matrix arithmetic may change draft IDs, which the target verifier
checks normally. Equal drafter arithmetic is not a requirement. No GPU-resident
FC/output-normalization rewrite is included.
