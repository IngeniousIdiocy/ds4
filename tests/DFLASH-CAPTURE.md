# DFlash serial capture SDN discriminator

The experimental serial capture path reuses the routed-slot shared-down/HC
epilogue on tapped sparse layers.  It writes the normal four-stream HC output
and the DFlash mean collapse in the same dispatch.  The existing standalone
`kernel_dsv4_hc_weighted_sum` path remains the default and the fallback.

`DS4_DFLASH_CAPTURE_SDN_FUSE=1` enables the candidate for one-row, armed,
tapped layers that actually take the routed-slot SDN tail.  Batched verifier
capture, non-slot tails, declined fusions, and failures use the standalone
collapse.  `DS4_GLM_EXACT=1` suppresses the candidate.

Use a real-input comparison before any default change:

```sh
DS4_DFLASH_CAPTURE_COMPARE=1 DS4_DFLASH_STATS=1 \
  ./ds4 -m TARGET.gguf --dflash DRAFTER.gguf \
  --dflash-mode conservative --temp 0 -p 'PROMPT' -n 2
```

The one-shot vehicle restores the same pre-token KDA and DSA-tail state before
each arm, then evaluates the same token at the same position with capture off,
standalone capture, and fused capture.  It compares every final HC word, every
logit, the serialized KDA plus DSA-tail state, and all five tap feature rows.
The ordinary standalone evaluation runs after the vehicle and is the state
retained by the session.

For alternating component timing, set the number of three-arm trials:

```sh
DS4_DFLASH_CAPTURE_BENCH=4 DS4_DFLASH_STATS=1 \
  ./ds4 -m TARGET.gguf --dflash DRAFTER.gguf \
  --dflash-mode conservative --temp 0 -p 'PROMPT' -n 2
```

Every interval is printed.  Trial zero is marked `cold=1` and remains in the
`all_mean_ms` result; only `warm_mean_ms` excludes it.  Later trials rotate arm
order.  These are repeated single-token component measurements, not generation
throughput.  Run the whole request separately with only
`DS4_DFLASH_CAPTURE_SDN_FUSE=1` after the exact comparison passes.
