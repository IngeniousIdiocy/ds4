# Model backing lifetime regression

Based on integration `b723dfa7594dbadc6480cf2872751ae22e551754`.

`newBufferWithBytesNoCopy` wraps caller-owned memory. The backend passes a nil
deallocator, so retaining the Metal buffer does not transfer ownership of the
engine's mmap. Apple describes the optional deallocator as the callback through
which the application releases that allocation when the buffer is deallocated:
[MTLDevice no-copy buffer documentation](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)).

The old engine close unmapped target/MTP/vision storage before backend cleanup.
Cleanup could subsequently commit a pending batch, then release residency and
no-copy views. Successful DFlash close also omitted the drafter model entirely.
A rejected drafter could be unmapped after map registration partially succeeded.
These are lifetime defects; this change does not establish any cause for earlier
file-cache accounting changes or performance variation.

The new ordering is:

1. Join CPU workers; stop Metal keepalive and TP service producers. TP shutdown
   releases pending gates, then drains while its poll/slab/check buffers exist.
   Main-queue keepalive waits its last submitted command before its thread exits.
2. Drain active/pending command buffers without initializing Metal on an early
   engine-open failure. Keep engine workspace, bank, draft and mapped backing
   alive, including with `DS4_METAL_UNRETAINED_COMMAND_BUFFERS` present.
3. Free engine-owned workspace/TP/draft buffers. Backend cleanup joins pread
   workers and releases residency, model views, exact-view caches and resources.
4. Close all four model mappings/file descriptors/metadata, including a disabled
   drafter whose registration was attempted. Pre-registration rejection can
   still close immediately. Partially allocated TP tables and draft pool/seed
   storage are also released.

Callers must already have freed sessions and stopped concurrent calls. This
patch does not add concurrent-close safety or certify arbitrary engine reopen
across devices; other process-global kernel scratch remains outside this fix.
CUDA workspace-before-backend-cleanup ordering is preserved; mapped backing now
closes after its backend cleanup too. Metal TP is source-audited, not GPU-tested
by the CPU fixture.

Run the bounded CPU regression (no Metal runtime/model load):

```sh
python3 tests/test_model_teardown.py
```

It compiles the production `ds4_engine_close` and `model_close` bodies against
lifetime-checking mocks, using real small mappings and file descriptors. Both
Metal-mock and CPU-only branches cover early failure, all auxiliary models,
rejected registered draft, successful draft and failed TP bind. The source
checks additionally pin Metal drain/release order, keepalive completion and
post-registration rejection ownership. `--source-only` avoids compilation.

After the coordinated final build, root should exercise normal server shutdown
with a loaded drafter and this tiny backend-only teardown vehicle:

```sh
make tests/test_metal_teardown   # build only; does not run the GPU
./tests/test_metal_teardown     # root-scheduled GPU window
```

The vehicle registers four 64 KiB synthetic file-backed mappings. It leaves
one batch submitted and another open, then prepares cleanup with unretained
references enabled. It verifies 256 exact one-hot matvec results through a host
pointer obtained before dispatch (no readback can mask a missing drain), frees
operands, cleans up the backend and finally unmaps/closes the four files. No
public model load or prompt preparation is needed; initialization dominates.

The CPU checks cannot prove Metal driver behavior or
substitute for those lifetime checks. No long-context performance rerun is
required to test this shutdown-only change.
