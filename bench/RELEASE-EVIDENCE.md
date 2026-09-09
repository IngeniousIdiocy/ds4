# GLM-5.3 M3 Ultra release evidence

This ledger records the final build of the `glm53-m3ultra` branch (release commit
`524c8a1`, tree `3049855`). Compact receipts live under
[`bench/receipts/glm53-m3ultra/`](receipts/glm53-m3ultra/); their staging labels and
source-receipt hashes preserve the link to the full logs without machine-local paths.
The earlier `538c37c` and `b723dfa` receipts are retained below as history.

## Bound identities

| input | identity |
|---|---|
| upstream base | `9ab705347c1775e7599ede7eb81a6255ec7dccb5` (not rebased; upstream `main` has moved since) |
| upstream comparator | same commit plus [upstream-bench-patch.diff](receipts/glm53-m3ultra/upstream-bench-patch.diff) (counters and EOS override only); `ds4` sha256 `f0d7d735…`; [upstream-baseline.json](receipts/glm53-m3ultra/upstream-baseline.json): 62k 365.70 prefill / 23.75 decode t/s, 300k 316.77 / 21.64 t/s |
| release commit | `524c8a1c22c02af6ed48082bea56942045225284`; tree `30498558ff3cbdc18db0c1d0f33c172277889867` |
| compiled sources | the tree of `247801c`; the later commits change only `bench/` scripts and documentation |
| final `ds4` | sha256 `11fba9964aca7d11644c3cf272b3a2c626727c10dd8810418cbab3da7258e99c` |
| final `ds4-server` | sha256 `511b7a9ae0951930ae20e7f76c22bc207e3f1466b58a26b9b20414134ed54ec5` |
| target GGUF | 185,299,232,064 bytes; sha256 `828f413cae8ceee74796606814295e00e04c1c7b151aca9a3c5c75db32cb73e0` |
| BF16 drafter (converted, not shipped) | 2,342,595,168 bytes; sha256 `a4bbfbd9e5db62ea31c5cde0bab38a4f9be11a8005dfd078f0d455bb630d66a9` |
| Q8_0 drafter (operator-quantized, not shipped) | 1,438,198,656 bytes; sha256 `9ea8a7387cb0429c657a4e504fc8779385109c20a8732fdb5cc8068ff3f54c75`; recipe in `docs/DFLASH_GLM53.md` section 9 |
| 62k needle prompt | 62,174 tokens; sha256 `e15ce96e009e4684d7006be75e6c24d2f8bfabf0a1ebacf9bc06a850034a6b9b` |
| 300k prompt | 300,000 tokens; sha256 `7a66f69f497d09e7c5c8dbe4295b8956b46e7874d03b7209c48b555bfc7efc9c` |

## Final artifact ledger (2026-09-09)

Native rows are bare defaults: no `DS4_`/`MTL_` variables, serial, no drafter loaded,
`DS4_GLM_IGNORE_EOS=1` to fix the 2,048-token horizon, GPU sampled idle before launch.

| gate | final result | receipt / scope |
|---|---|---|
| native 62,174-token prefill | **550.37 t/s**; all 42 routed layers banked, no refusal | [final-native62.json](receipts/glm53-m3ultra/final-native62.json); guarded policy resolved bank, fused command buffer, eight-layer pipeline, untracked model views |
| native 62,174-token serial decode | **38.067237857 t/s**; `n_generated=2048`, `n_decode_eval=2047`, `stop=predict_limit`; 53.799543 s | same receipt; output 8,969 bytes byte-identical to the retained `BASE62-v3` controls and the `538c37c` receipt |
| native 300,000-token serial decode | **37.387743210 t/s**; `n_generated=2048`, `n_decode_eval=2047`, `stop=predict_limit`; 54.77731 s | [final-native300.json](receipts/glm53-m3ultra/final-native300.json); output 9,185 bytes byte-identical to the retained v3+`xr8` block and the `538c37c` receipt |
| native 300,000-token prefill | **473.92 t/s**; five bank groups, no refusal | same receipt |
| DFlash fixtures, Q8_0 drafter, 512 tokens each | serial / conservative / speculative: SQL **40.91 / 62.19 / 63.34**, JSON **40.68 / 49.39 / 54.24**; conservative on prose 39.90 (-2.0%), chat 39.49 (-3.0%), mixed prose+SQL 39.97 (-1.8%), 24-token restored 300k prefix 36.10 (-5.4%); every DFlash output byte-identical to its serial arm | [dflash-fixtures.json](receipts/glm53-m3ultra/dflash-fixtures.json); complete CLI generation intervals; short fixtures, not a workload average |
| real-agent screen (selection evidence) | conservative **39.73 vs serial 38.09** output t/s (+4.3%) over 32 randomized requests; leave-one-out +3.2% to +5.6% | `docs/DFLASH_GLM53.md` section 7; measured on the accepted configuration before this port, same controller and drafter |
| drafter fault latch | **passed**: injected drafter failure in conservative and speculative servers; faulted request and same-session reuse byte-identical to the serial control; safe drain, session latch | [dflash-runtime.json](receipts/glm53-m3ultra/dflash-runtime.json) |
| server lifecycle (conservative) | **7/7 passed**: verified cancel, same-slot reuse after cancel and after natural EOS, history invalidation, positive-temperature serial gate, usage accounting | same receipt |
| ignore-EOS, same server, both public profiles | **passed**: 64-token `ignore_eos` pairs in serial, conservative and speculative all reach the exact cap and match the serial permitted-token reference byte for byte (conservative declined every post-EOS proposal; speculative verified 48 blocks) | same receipt |
| CPU unit tests | DFlash budget/adaptive/retry/clock/entry/windowed/selector-confidence/fault/confidence/mode/seed tests and `ds4_test --server` (including the two 2026-09-08 server-fix tests) pass; `tests/test_dflash_sdpa` (GPU) passes with split attention 13x faster than the single-pass kernel at 2,047 rows | build log; no warnings |

A preceding 62k attempt on the same binary measured 549.92 prefill t/s with the GPU
sampled contended (WindowServer active) before launch; it is retained in the receipt and
was not selected. The 62k and 300k outputs match the `538c37c` receipts byte for byte:
the DFlash and server changes do not touch the native serial path.

The public reproducer is [`reproduce-glm53-native.sh`](reproduce-glm53-native.sh). It
scrubs inherited tuning variables and does not rehash the 185 GB model.


## Historical `538c37c` final artifact (2026-09-06)

The previous public candidate, retained for lineage; the two native outputs above match these byte for byte.

| gate | final result | receipt / scope |
|---|---|---|
| native 62,174-token prefill | **550.27 t/s**; all 42 routed layers banked, no refusal | [final-native62.json](receipts/glm53-m3ultra/hist-538c37c-native62.json); bare guarded policy resolved bank, fused command buffer, fixed eight-layer pipeline and untracked model views |
| native 62,174-token serial decode | **37.868944385 t/s**; `n_generated=2048`, `n_decode_eval=2047`, `stop=predict_limit` | same receipt; 54.081254 s; fixed horizon with EOS ignored |
| native 300,000-token serial decode | **37.187023218 t/s**; `n_generated=2048`, `n_decode_eval=2047`, `stop=predict_limit` | [final-native300.json](receipts/glm53-m3ultra/hist-538c37c-native300.json); 55.072975 s; fixed horizon with EOS ignored |
| native 300,000-token prefill | **473.64 t/s**; five bank groups, 210 bank expansions, no refusal | same receipt; bare guarded policy |
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
and matched the retained v3+`xr8` and `b723dfa` blocks byte for byte. 
## Earlier capability and quality evidence

| gate | retained evidence |
|---|---|
| native prefill at 62,174 tokens | **550.72 t/s**, forced bank + fused command buffer + eight-layer pipeline; `ASTRAL-BANK-PIPELINED62-20260906T194550Z` |
| native serial decode at 300,000 tokens | **37.2993996 t/s**, 2,048 generated / 2,047 evaluated, prediction-limit stop; `BASE300-astral-b723-pipeline-20260906T195448Z` |
| prefill at 300,000 tokens | **473.75 t/s**, five admitted bank groups and no refusal; same receipt |
| DFlash three-mode fixed horizon | serial **38.5850**, conservative **47.2064**, speculative **60.7875 t/s**; 8,192 generated and 22,036 equal output bytes per arm; [dflash-three-mode.json](receipts/glm53-m3ultra/dflash-three-mode.json) contains the exact prompt |
| task outcome screen | v3 candidate versus upstream: **77/77** in both arms; 22/23 byte-identical, one wording difference; [taskcheck-quality.json](receipts/glm53-m3ultra/taskcheck-quality.json) and [subitems TSV](receipts/glm53-m3ultra/taskcheck-subitems.tsv) |
| 100-prompt reference NLL (release criterion) | final build `999f510` vs upstream `9ab7053`, same file, one window: upstream **0.300804038** (90/100, lcp 9.48; equals upstream's published QA figure), defaults **0.300766166** (90/100, lcp 10.29), `DS4_GLM_EXACT=1` **0.300759677** (90/100, lcp 9.48); both branch arms below the pin, `compare_1k.py` criterion 2 PASS for both; [fidelity-100.json](receipts/glm53-m3ultra/fidelity-100.json), TSVs under [bench/fidelity](fidelity/) |
| E6 long-context NLL screen (diagnostic) | token-weighted NLL 1.270792 vs upstream 1.268859, delta +0.001933, casewise SE 0.001757 (1.1 SE), 6 of 12 cases better; re-run on `999f510` reproduced both Sep 6 TSVs byte for byte; [e6-quality.json](receipts/glm53-m3ultra/e6-quality.json), [candidate TSV](receipts/glm53-m3ultra/e6-candidate.tsv), [upstream TSV](receipts/glm53-m3ultra/e6-upstream.tsv) |

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
