# Experiment 17

**Date:** 2026-03-19T19:26:29.536646+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5349
**Artifact size:** 14,490,562 bytes
**Model params:** 21485864
**Last step:** 311
**Pre-quant val_bpb:** 1.5319
**Quantization gap:** 0.0030
**Eval time:** 16075 ms
**Peak memory:** 14180 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5349)
**Propose time:** 0.0s
**Train time:** 332.5s

## Change
Reduce muon_momentum_warmup_steps from 500 to 200 to let the Muon optimizer reach full momentum (0.95) by step 200 instead of step 500. With only ~377 steps in the 180s proxy, the old 500-step warmup meant the optimizer NEVER reached its target momentum (only 0.925 at the end). This is the same class of fix as the successful warmdown_iters reductions — adapting a schedule designed for 20K iterations to work properly on the short-horizon proxy. Zero impact on model size, parameters, or step time.

## Diff from previous best
+3 lines / -3 lines (vs current best)
