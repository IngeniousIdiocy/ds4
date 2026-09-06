# Compile-only routed failure and re-prime screen

This test complements the production-binary cancellation screen. It injects a
deterministic host failure after encoding real routed work; it does not emulate
a GPU device error, allocation failure or failed state restoration.

Build, without running the GPU:

```sh
make tests/ds4_server_sc_failure
python3 tests/run_superchunk_failure.py --self-test
```

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

1. Fresh server: short prompt A produces the baseline assistant content and
   consumes no superchunk group.
2. Long prompt B: the initial ordinary chunk completes 4,096 tokens. Group 1
   then completes 32,768 tokens (positions 4,096–36,863). In group 2, the eighth
   real banked routed dispatch is encoded before the test sets `rc=FAILED`.
   Normal code drains, restores and invalidates the caller's nonempty
   checkpoint at 36,864. The driver requires exactly eight banks and no bank refusal.
3. Short prompt A again: the same session must take the re-prime branch from
   invalid state and successfully reset KDA state. Its complete assistant
   content must be byte-identical to the fresh baseline; JSON IDs/timestamps
   are not compared.
4. The owned server must exit cleanly after SIGTERM.

The historical `needle-64k.txt` prompt has approximately 62,174 model tokens.
At minimum=maximum=32,768 its unpadded remainder cannot form group 2. The driver
appends 4,096 deterministic ` 17` units by default; `--padding 0` accepts an
already suitable prompt. Eligibility is established by actual receipts,
requiring the injected group to start at 36,864 and span 32,768 (at least 69,632
total prompt tokens). A short or
oversized prompt fails the screen, even if the server responds healthily.

Expected GPU work is an ordinary 4k prefill, one full 32k group, and eight routed layers of another
32k group and two tiny generations, after one engine startup. This is a
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

Reanalyze those unchanged receipts on the CPU; save the output separately:

```sh
python3 tests/run_superchunk_failure.py --self-test
python3 tests/run_superchunk_failure.py --receipts \
  /path/to/ASTRAL-FINAL-failure-20260906T204052Z/vehicle
```

This correction changes only the comparator/documentation. No engine hook,
binary, saved receipt or GPU run changes. Reanalysis also requires zero cached
prompt tokens in baseline and recovery responses, and retains the original
failure result in its output.
