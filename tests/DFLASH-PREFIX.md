Accepted-prefix restoration
===========================

The target verify now snapshots the complete recurrent state at every row.
Each step is the KDA conv/recurrent regions in forward layer order, followed
by the DSA indexer K+gate tails in forward layer order. On the 45-layer trunk,
the existing 152,633,344-byte KDA step gains 45,056 bytes of DSA tails: 360,448
additional bytes for eight steps. The pre-block 152,678,400-byte backup remains
available for failure recovery and restore+replay.

Before each indexer pool update, a small kernel reconstructs all eight tails
from the untouched pre-block tail and this layer's raw K/gate projections.
For each prefix/slot it selects the last prefix row with that slot modulo 4.
Slot 3 retains its old value because the serial producer sends a completing
row directly to the compressed pool without writing the tail. This includes
inactive slots so rejected rows cannot survive there as diagnostic noise.

The graph restores row `committed - 1` on partial acceptance. Capture cursors
must cover the expected KDA and tail bytes; a missed/incompatible capture
selects the backup+replay path. Instrumentation is armed/disarmed within one
session's cycle, and existing failure/cancellation invalidation still applies.

Compressed keys and compact KV are addressed by absolute position. The serial
indexer reads `visible / 4` pooled rows; `glm_indexer_batch_visible_rows` masks
each batched query the same way. Payload live counts derive from the committed
checkpoint. Future pool/KV entries are hidden, and the next genuine producer
overwrites them before they become visible. The incomplete tail is restored,
including when the verify crossed a pool boundary that acceptance did not.
Dense-cache validity advances over committed rows only.

`DS4_DFLASH_FORCE_REPLAY=1` now forces serial replay even after full acceptance.
It reconstructs serial state on the accepted IDs. Batched target argmax itself
can differ from serial, so this is not a universal identical-generation claim.
The default path retains causal batched arithmetic for all accepted prefixes;
serial-versus-batched numerical quality needs separate measurement.

The adaptive controller also seeds its serial EMA from successful feature-
refresh evaluations, without adding any model work. Conditioning reset clears
both timing EMAs because a new/restored request may have a different depth.

Checks
------

CPU-only, no model/backend initialization:

```
make dflash-prefix-test
python3 tests/dflash_prefix_fixtures.py
python3 tests/dflash_rejection_fixtures.py
```

The small C test drives production source-index and rollback functions for
all start residues and acceptance lengths, retains sentinel pre-block slots,
varies rejected suffix rows, and follows restoration through pool completion.
The Python fixtures include changed-state and missing-inventory negatives.

GPU owner, from the built worktree, with MODEL and DFLASH set to model files:

```
DS4=./ds4 OUT=/tmp/dflash-prefix sh tests/dflash_prefix_harness.sh
DS4=./ds4 OUT=/tmp/dflash-replay sh tests/dflash_rejection_harness.sh
```

The prefix harness runs four short processes: serial ID reference, suffix A,
repeat A, suffix B. Its one 8-row block starts at residue 1 and commits 5 rows;
the entire rejected suffix changes, including its first row. Each changed
draft is guaranteed to differ from the recorded reference; an unexpected
acceptance count still refuses the run. It checks the complete
canonical state, live compact KV and pooled-key inventory, logits, and serial
continuation across the next pool boundary. A/A and A/B observations remain
separate. Identical controls establish suffix invariance at these tested
frontiers, rather than universal causal correctness;
differences require diagnosis and are not automatically accepted or blamed on
contamination. No equality to serial batched arithmetic is assumed here.
The replay harness retains its prior boundary coverage and now checks full
acceptance through replay as well. Both harnesses are untimed diagnostics.

Then rerun the existing short JSON generation and one cached-depth fallback
check with `DS4_DFLASH_STATS=1`: confirm nonzero `serial`, updated `be`, partial
`rollback=stepsnap`, and no incomplete-capture warning. Measure throughput
without state/cache dump instrumentation. No broad validation matrix is needed
to decide whether this removes the replay cost.

Scope: GPU kernels, causal suffix control, serial quality, and throughput must
be measured by the GPU owner; a host build and CPU tests cannot certify them.
