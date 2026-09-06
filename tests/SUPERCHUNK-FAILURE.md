# Compile-only routed failure and re-prime screen

This test complements the production-binary cancellation screen. It injects a
deterministic host failure after encoding real routed work; it does not emulate
a GPU device error, allocation failure or failed state restoration.

Build, without running the GPU:

```sh
make tests/ds4_server_sc_failure
python3 tests/run_superchunk_failure.py --self-test
```

An existing `tests/ds4_server_sc_failure` from the final 538c37c build can be
reused. Comparator v3 changes only this Python driver and documentation; it
requires no engine/server/Metal rebuild.

The target recompiles only `ds4.c` into `tests/ds4_sc_failure.o`, with
`DS4_TEST_GLM_SC_FAIL_GROUP=2`, then links the normal server and Metal objects.
It deliberately does not define `DS4_TEST_HOOKS`, which enables unrelated test
behavior elsewhere. The production `ds4.o` and executable are unchanged. All
fault branches, counters and receipts are inside the dedicated macro guard.

Root schedules the runtime screen while holding the existing GPU lock:

```sh
python3 tests/run_superchunk_failure.py \
  --model /path/to/GLM-5.3-Flash-Q4_K.gguf \
  --fault-prompt /path/to/needle-64k.txt \
  --lock-script /path/to/gpulock.sh \
  --lock-token "$TOK" --port 8099 --out /absolute/path/to/new-receipt-directory
```

The driver verifies lock ownership before creating its server. It never stops
an existing listener, never releases the caller's lock, and only terminates
the process it created. No disk KV-cache directory or batched-session option
is supplied. It uses serial decode and the shared/fused/pipelined expert bank
with 8,192-token slices, setting both superchunk maximum and minimum to 32,768.
The four phases are:

1. Fresh server: short prompt A requests integers 1 through 100, one per line.
   Greedy decoding has a 128-token cap and normal EOS behavior; `IGNORE_EOS` is
   absent. Native server usage must report at least 64 generated tokens and a
   positive prompt count. An insufficient baseline fails before the expensive
   fault prefill. This short prompt consumes no superchunk group.
2. Long prompt B: the initial ordinary chunk completes 4,096 tokens. Group 1
   then completes 32,768 tokens (positions 4,096–36,863). In group 2, the eighth
   real banked routed dispatch is encoded before the test sets `rc=FAILED`.
   Normal code drains, restores and invalidates the caller's nonempty
   checkpoint at 36,864. The driver requires exactly eight banks and no bank refusal.
3. Short prompt A again: the same session must take the re-prime branch from
   invalid state and successfully reset KDA state. Its complete assistant
   content must be byte-identical to the fresh baseline; JSON IDs/timestamps
   are not compared. Both responses must report the same native completion
   count in 64..128, positive prompt usage, consistent total counts and zero
   cached prompt tokens. Missing, noninteger or inconsistent usage fails.
4. The owned server must exit cleanly after SIGTERM.

The historical `needle-64k.txt` prompt has approximately 62,174 model tokens.
At minimum=maximum=32,768 its unpadded remainder cannot form group 2. The driver
appends 4,096 deterministic ` 17` units by default; `--padding 0` accepts an
already suitable prompt. Eligibility is established by actual receipts,
requiring the injected group to start at 36,864 and span 32,768 (at least 69,632
total prompt tokens). A short or
oversized prompt fails the screen, even if the server responds healthily.

Expected GPU work is an ordinary 4k prefill, one full 32k group, and eight routed
layers of another 32k group plus two generations capped at 128 tokens, after
one engine startup. The retained fault prefill took approximately 80 seconds;
the two new generations add at most roughly 6–7 seconds at the previously
observed short-context rate. Allow approximately 90–100 seconds with startup,
subject to hardware variance. This is a
bounded correctness screen, not a throughput benchmark. The JSON result records
the precise exercised frontier; it cannot certify arbitrary later failures.

The CPU self-test rejects planted first-group faults, incorrect 32,768 frontiers,
an incomplete first group, fewer than eight banks, a retained-valid checkpoint,
a different recovery session or reset length, changed assistant
content and an empty baseline. It also checks that test logic is guarded out
of the production source. Runtime receipts remain required to establish actual
bank engagement, failure propagation and successful recovery.

## Comparator correction, v2

The original comparator incorrectly assumed group 1 began at position zero.
The retained `ASTRAL-FINAL-failure-20260906T204052Z` run shows the initial
4,096-token ordinary chunk, completed group 1 at 36,864, injected group 2 with
eight banks, `FAILED` with checkpoint 36,864 invalid, same-session re-prime
from that invalid checkpoint, identical `healthy` assistant bytes and clean
server exit. Its original `result.json` remains a comparator failure.

Comparator v2 (commit 186d064) reanalyzed those unchanged receipts successfully,
including zero cached prompt tokens and the original failure result in its
output. No engine hook, binary, saved receipt or GPU run changed.

## Recovery sensitivity, v3 — runtime pending

The retained v2 receipt compared only `healthy`: seven bytes and one generated
token. Its measured scope remains unchanged. It does not establish the stronger
64-token recovery requirement. V3 uses a substantial deterministic-output
request and positive native server counts in both arms; it never counts words,
characters or tokens from a separate tokenizer as a substitute.

The 64- and 128-token CPU controls pass. Negative controls cover either arm
below 64 or above 128, unequal completion counts, zero prompt usage, missing
usage, booleans/floats/NaN in counts, inconsistent totals and cached prompts,
in addition to the existing frontier and content mismatches. The saved
recovery prompt, its hash, the requested cap and both measured completion
counts are recorded in the new artifacts.

`--receipts` uses the current v3 criterion and must reject the old one-word
receipt. A new root-scheduled runtime screen after the native gates is required
to claim v3 coverage. No such runtime measurement is claimed by this update.
