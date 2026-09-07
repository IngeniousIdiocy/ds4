# Bounded Spec-Bench subset

`reproduce-spec-bench-subset.py` runs a frozen, category-balanced subset of the
upstream Spec-Bench dataset. It is intended to expose DFlash behavior between
small hand-written fixtures and a full benchmark run. It is not a full
Spec-Bench score.

The driver requires the upstream repository at commit
`fd2c1cd7d2201ef71db4c5f4e455008f017967bf`. It selects the first 12 question
rows, in stable file order, from each of `mt_bench`, `translation`,
`summarization`, `qa`, `math_reasoning`, and `rag`. The 12 selected MT-Bench
questions each have two turns; the remaining 60 questions have one. The fixed
scope is therefore 72 questions, 84 sequential turn requests per mode, and 252
requests across serial, conservative, and speculative modes. Each turn is
greedy, permits natural EOS, and has a maximum of 256 generated tokens.

The dataset text remains in the external checkout. `subset-manifest.json`
binds every source row and turn by index, ID, byte count, and SHA-256 without
copying its text into this repository. Live raw request and trace artifacts do
contain the prompts needed to audit a run.

Use a new output directory. With the shared GPU lock:

```sh
python3 bench/reproduce-spec-bench-subset.py \
  --source-root /path/to/ds4-source \
  --server /path/to/ds4-server \
  --model /path/to/GLM-5.3-Flash-Q4_K.gguf \
  --draft-model /path/to/GLM-5.3-Flash-DFlash2.gguf \
  --spec-bench-root /path/to/Spec-Bench \
  --source-license-metadata /path/to/source-dataset-licenses.json \
  --output /new/evidence/directory \
  --gpu-lock-script /path/to/gpulock.sh
```

Add `--dry-run` to validate identities and freeze the plan without taking the
lock or starting a server. If a compatible lock helper is unavailable, the
caller must already hold exclusive GPU ownership and pass
`--i-own-exclusive-gpu`.

Before the full subset, `--preflight-smoke` runs a separate functional check:
one pinned two-turn MT-Bench row, one pinned single-turn QA row, and two
immediately repeated copies of the original three-token factual-EOS fixture in
each mode. This is 15 requests and is always labeled as smoke evidence with no
subset performance claim. A live smoke also requires `--telemetry-helper`;
the helper samples idle GPU state after lock preflight and before each server
starts. It is never called while the model child is live.

The smoke and full subset leave the conservative immediate-entry default unset.
The repeated three-token factual requests must exercise verified partial-EOS
accounting in both public DFlash modes. They additionally require a real
width-seven full-block verification in speculative mode. `--min-serial-tokens
3` is an explicit conservative serial-entry diagnostic control; speculative
always uses the public full-block policy and ignores that control.

One server process is started per mode and handles all 84 turns for that mode,
so the target and drafter load once per mode. The driver scrubs inherited
`DS4_` and `MTL_` variables. It enables statistics, sets a conservative
entry-gate variable only for the explicit diagnostic control, and records the
selected binary's policy and profile receipts. Conservative retains its
confidence, economics, retry, and meter policy. Public speculative uses full
width without confidence, economics, retry, or the loss meter.
`--proposer-head-nt4` enables that opt-in only for the conservative server.

Every declared turn gets one attempt. The driver retains the raw request,
response, trace, server-log segment, status, and parsed receipt. It continues
with independent questions after a validation failure and never retries or
chooses a favorable attempt. A failed first turn blocks only its dependent
second turn and invalidates the aggregate.

The report keeps three timing boundaries separate:

- Server end-to-end time is the native trace timer from before prefill through
  generation.
- Decode-only time is the final native decode-progress timer, which starts
  immediately after prefill.
- Client wall time is the monotonic duration of the HTTP request.

For each boundary it reports pooled throughput, mean per-question throughput,
matched total elapsed ratios, and matched throughput ratios. The two ratios are
separate because natural-EOS output lengths can differ.

Accepted draft tokens per verify use runtime acceptance receipts. Conservative
uses its policy accounting ledger; accepted-count verifier-cap adaptation is
off by default. Public speculative emits the same
call/ACK/totals structure for accounting around its full-block verifier, while
the frontend ACK supplies the actual consumed prefix. Committed tokens per
verify therefore use the real `consumed` count from each verified call in both
modes, including terminal partial consumption. Historical cycle-only receipts
remain parseable but are rejected as public-profile evidence; only those old
receipts require the conventional accepted-plus-anchor estimate. Internal
counterfactual net time remains separate from the paired measured slowdown.

When EOS appears inside a returned DFlash block, the target has already
evaluated the whole returned block while only the visible prefix is consumed.
The validator requires native forward positions to equal adaptive `returned`,
and visible generated tokens to equal adaptive `consumed` for a natural stop.
It records `returned - consumed` as the unconsumed evaluated EOS suffix.

The driver leaves disk KV caching unconfigured. It requires zero cached prompt
tokens on the first turn of each independent question and checks the cache-read
count against both trace and API usage. A second MT-Bench turn may reuse that
mode's live conversation, and includes that mode's actual first answer. Later
prompts can therefore differ across modes; the report retains their rendered
prompt identities and reused-token counts. These are measured request timings,
with no hypothetical speed attributed to an independently warmed prefix.

The source checkout's Apache-2.0 LICENSE and case-sensitive `Readme.md` are
hash-bound in the identity receipt. That repository license is not a claim that
all underlying source-dataset texts have the same license. Dataset text remains
external; its upstream source attribution and terms remain applicable.
The required reviewed source-license JSON is hash-bound and copied into every
run directory; it preserves category-specific license evidence and explicitly
unresolved row-level lineage.

Interrupted and failed-start runs retain completed attempt records and explicit
unattempted records for the rest of the active declared schedule. A failed
first turn blocks its dependent turn. Each successful trace must contain the
exact JSON request sent by the client, including the mode-local conversation.
