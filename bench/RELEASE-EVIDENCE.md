# GLM-5.3 M3 Ultra release evidence

This ledger separates the final `538c37c` artifact from the earlier `b723dfa`
capability build. Compact final receipts live under
[`bench/receipts/glm53-m3ultra/`](receipts/glm53-m3ultra/); their staging labels and
source-receipt hashes preserve the link to the full logs without machine-local paths.

## Bound identities

| input | identity |
|---|---|
| upstream/release base | `9ab705347c1775e7599ede7eb81a6255ec7dccb5` |
| final compiled source | `538c37c0455a58df5024bec58be99a4ecfef58a7`; tree `f4141a9f7aa258e03f3f8f8145e65e052091402c` |
| curated source-equivalent checkpoint | `26e454fbd93ee545eec1a928ab8f2a0d47dc23cd`; empty tree diff against `538c37c` |
| final `ds4` | sha256 `79dd8f58f1d74d6f43a6ce3dbd52c60d5eda0119ab821d37d72f52b6a17d027b` |
| final `ds4-server` | sha256 `9347f0d9d50050e627018621ba7d6bb9b973116d8a0554463155ce72d5dc6429` |
| target GGUF | 185,299,232,064 bytes; sha256 `828f413cae8ceee74796606814295e00e04c1c7b151aca9a3c5c75db32cb73e0` |
| 62k needle prompt | 62,174 tokens; sha256 `e15ce96e009e4684d7006be75e6c24d2f8bfabf0a1ebacf9bc06a850034a6b9b` |
| 300k prompt | 300,000 tokens; sha256 `7a66f69f497d09e7c5c8dbe4295b8956b46e7874d03b7209c48b555bfc7efc9c` |

The runtime checkout included `d1838d7`, a Python comparator and documentation change
made after the final build. It did not change compiled inputs. The curated `26e454f`
checkpoint preserves the exact compiled tree. A later two-line comment cleanup changed
the `ds4_metal.m` source-file digest while preserving line count and producing
byte-identical preprocessor output under the production Objective-C flags; see
[comment-preprocess-proof.json](receipts/glm53-m3ultra/comment-preprocess-proof.json).
All other later changes are tests, documentation, scripts and receipts.

## Final artifact ledger

| gate | final result | receipt / scope |
|---|---|---|
| native 62,174-token prefill | **550.27 t/s**; all 42 routed layers banked, no refusal | [final-native62.json](receipts/glm53-m3ultra/final-native62.json); bare guarded policy resolved bank, fused command buffer, fixed eight-layer pipeline and untracked model views |
| native 62,174-token serial decode | **37.868944385 t/s**; `n_generated=2048`, `n_decode_eval=2047`, `stop=predict_limit` | same receipt; 54.081254 s; fixed horizon with EOS ignored |
| native 300,000-token serial decode | **37.187023218 t/s**; `n_generated=2048`, `n_decode_eval=2047`, `stop=predict_limit` | [final-native300.json](receipts/glm53-m3ultra/final-native300.json); 55.072975 s; fixed horizon with EOS ignored |
| native 300,000-token prefill | **473.64 t/s**; five bank groups, 210 routed layers, no refusal | same receipt; bare guarded policy |
| guarded boundary | **556.73 AUTO vs 540.64 bank-off t/s** at 33,148 native tokens; same 77 bytes | [final-bank-boundary33.json](receipts/glm53-m3ultra/final-bank-boundary33.json); fixed AUTO-then-OFF pair supports 32768 but does not locate a crossover |
| startup/server affected runtime | **16/16 passed** | [final-runtime.json](receipts/glm53-m3ultra/final-runtime.json); mode precedence/bypass, model aliases, cancellation/reuse, conservative credit, speculative stop and natural EOS |
| pipelined routed disconnect cancellation | **5/5 passed**; 7 pipeline flushes, 8 routed batches, state restored, healthy reuse byte-identical to fresh control | same runtime receipt; disconnect occurred after queued routed work |
| connected-client SIGTERM | **4/4 passed**; prefill restored before the client socket closed | same runtime receipt; the client remained connected through restoration |
| Metal model-view teardown | **passed** with four maps, pending/open batches and 256 results | same runtime receipt |
| routed failure/re-prime | comparator v3 **passed**: fault after 8 routed layers/banks, statuses 200/500/200, baseline/recovery 128 native tokens and 183 identical bytes | same runtime receipt; zero cached prompt tokens; same-session re-prime; server exit 0 |
| JSON512 observation | **40.632779352 t/s**; `n_generated=512`, `n_decode_eval=511`, natural EOS enabled | [final-json512.json](receipts/glm53-m3ultra/final-json512.json); output matched prior JSON512 bytes, but the task was truncated at the prediction limit |

The 62k output was 8,969 bytes, sha256
`1bb479580d47c8863438e01585b5db9d14e532a8da207a635a3d57686eae7df5`,
and matched all three retained `BASE62-v3-20260906T133801Z` controls byte for byte.
That is evidence for this fixture and lineage, not a universal identity requirement.
The 300k output was 9,185 bytes, sha256
`2c019d7749ac60449bc3fbb6999bad4c1a03be10e7816e6d7403722ab874b1dc`,
and matched the retained v3+`xr8` and `b723dfa` blocks byte for byte. Both native hard
gates therefore pass on the final binary: 550.27 t/s at 62k prefill and 37.187023218
t/s at 300k serial decode.
The public reproducer is [`reproduce-glm53-native.sh`](reproduce-glm53-native.sh). It
scrubs inherited tuning variables and does not rehash the 185 GB model.

## Earlier `b723dfa` capability receipts

| gate | `b723dfa` evidence |
|---|---|
| native prefill at 62,174 tokens | **550.72 t/s**, forced bank + fused command buffer + eight-layer pipeline; `ASTRAL-BANK-PIPELINED62-20260906T194550Z` |
| native serial decode at 300,000 tokens | **37.2993996 t/s**, 2,048 generated / 2,047 evaluated, prediction-limit stop; `BASE300-astral-b723-pipeline-20260906T195448Z` |
| prefill at 300,000 tokens | **473.75 t/s**, five admitted bank groups and no refusal; same receipt |
| DFlash three-mode fixed horizon | serial **38.5850**, conservative **47.2064**, speculative **60.7875 t/s**; 8,192 generated and 22,036 equal output bytes per arm; [dflash-three-mode.json](receipts/glm53-m3ultra/dflash-three-mode.json) contains the exact prompt |
| task outcome screen | **77/77** in both arms; 22/23 byte-identical, one wording difference; [taskcheck-quality.json](receipts/glm53-m3ultra/taskcheck-quality.json) and [subitems TSV](receipts/glm53-m3ultra/taskcheck-subitems.tsv) |
| E6 quality margin | **unmet**: token-weighted NLL 1.270792 vs upstream 1.268859, delta +0.001933; provisional +0.0005 criterion not met; [e6-quality.json](receipts/glm53-m3ultra/e6-quality.json), [candidate TSV](receipts/glm53-m3ultra/e6-candidate.tsv), [upstream TSV](receipts/glm53-m3ultra/e6-upstream.tsv) |

The DFlash result is one favorable repetitive SQL fixture. Every arm stopped at the
8,192-token horizon during tuple 483 of a requested 2,000, so it is neither a
completed-task result nor a representative-workload average. Matching bytes in that
run do not create a universal quality or batching contract.

## Evidence limits

The exact-mode public-artifact diagnostic, vision revalidation, and broader upstream
storage configurations remain
documented limits. Scheduling and default wiring do not justify mechanically repeating
unchanged 25-minute quality matrices; any future numerical change must use the fidelity
gates in this directory.

Routine reproduction binds the 185 GB model by the pinned known digest plus file size
and local file identity, and hashes the small prompts, binaries and receipts. Rehash the
entire model only when its provenance is actually in question.
