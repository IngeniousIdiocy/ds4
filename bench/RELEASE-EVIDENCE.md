# GLM-5.3 M3 Ultra release evidence

This ledger separates the measured `b723dfa` capability build from the final rebuilt
release artifact. A staging receipt name identifies the retained evidence directory;
publication should copy the compact receipt into the release tree or replace the name
with a stable repository-relative link.

## Bound inputs

| input | identity |
|---|---|
| target GGUF | 185,299,232,064 bytes; sha256 `828f413cae8ceee74796606814295e00e04c1c7b151aca9a3c5c75db32cb73e0` |
| 62k needle prompt | 62,174 tokens; sha256 `e15ce96e009e4684d7006be75e6c24d2f8bfabf0a1ebacf9bc06a850034a6b9b` |
| 300k prompt | 300,000 tokens; sha256 `7a66f69f497d09e7c5c8dbe4295b8956b46e7874d03b7209c48b555bfc7efc9c` |
| capability source | `b723dfa7594dbadc6480cf2872751ae22e551754` |
| capability `ds4` | sha256 `31139ed7f231856d4ba5414dbd375c2dfcea3965dc1b7378a5e8e77e98fe74e1` |

## Receipt ledger

| gate | `b723dfa` evidence | final rebuilt artifact |
|---|---|---|
| native prefill at 62,174 tokens | **550.72 t/s**, all 42 layers banked; forced bank + fused layer command buffer + eight-layer pipeline; `ASTRAL-BANK-PIPELINED62-20260906T194550Z` | pending guarded-default build and 32k boundary screen |
| native serial decode at 300,000 tokens | **37.2993996 t/s**, 2,048 generated / 2,047 evaluated, `stop=limit`; `BASE300-astral-b723-pipeline-20260906T195448Z` | pending final binary receipt |
| prefill at 300,000 tokens | **473.75 t/s**, five bank groups admitted, none refused; same receipt | pending final binary receipt |
| DFlash three-mode fixed horizon | serial **38.5850**, conservative **47.2064**, speculative **60.7875 t/s**; 8,192 generated and 22,036 equal output bytes in each arm; `ASTRAL-THREE-MODES-20260906T192241Z` | pending affected runtime receipt |
| startup/server affected runtime | **16/16 passed**: explicit mode precedence and draft bypass, `/v1/models` aliases, bank cancellation then healthy reuse, speculative stop, natural EOS and healthy reuse; `ASTRAL-RUNTIME-20260906T195146Z` | pending focused rerun; early bank cancellation did not prove queued pipeline drain |
| task outcome screen | **77/77** in both compared arms; 22/23 byte-identical, one wording difference; `TASKCHECK-20260906T154355Z` | carried evidence; rerun only if the final change can affect numerics |
| E6 quality margin | **unmet**: token-weighted NLL 1.270792 vs upstream 1.268859, delta +0.001933; provisional +0.0005 criterion not met | disclose; no unchanged quality-matrix rerun solely for scheduling/default wiring |

The DFlash result is one favorable repetitive SQL fixture. Every arm stopped at the
8,192-token horizon during tuple 483 of a requested 2,000, so it is not a completed-task
or representative-workload speed claim. Matching bytes in that run are evidence about
that fixture, not a universal quality or batching contract.

## Final receipt requirements

The final receipt should record source and binary hashes, model stat and pinned known
digest, prompt digest, resolved startup policy, effective environment, generated and
evaluated counts, stop reason, prefill and decode times, bank admission/engagement, and
server event evidence. Avoid rehashing the 185 GB model during routine runs; bind it by
the pinned known digest plus file size and modification identity, and hash the small
prompts, binaries and receipt files.

The final affected checks are narrow: guarded profile resolution (including
`model_untracked`), the 32k bank boundary, native 62k prefill, one complete native
300k/2,048 serial block, startup/server lifecycle paths changed since `b723dfa`, and
shutdown cancellation during long prefill. Do not substitute the cached harness for the
native 300k gate or mechanically repeat unchanged broad quality matrices.
