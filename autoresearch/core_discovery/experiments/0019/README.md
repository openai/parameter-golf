# Experiment 19

**Date:** 2026-03-19T19:47:55.554483+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5348
**Artifact size:** 11,976,518 bytes
**Model params:** 17342176
**Last step:** 328
**Pre-quant val_bpb:** 1.5323
**Quantization gap:** 0.0025
**Eval time:** 15285 ms
**Peak memory:** 12106 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5348)
**Propose time:** 0.0s
**Train time:** 264.5s

## Change
Reduce warmdown_iters from 600 to 550 to test a midpoint between the current best (600, bpb=1.4967) and the too-aggressive 400 (bpb=1.5174). On the 180s proxy (~377 steps), this raises the initial effective LR from ~63% to ~68% of base, giving slightly more productive learning time at higher LR while still maintaining smooth end-of-training decay. On the full 600s run (~1061 steps), warmdown starts at step 511 (vs 461 at 600), meaning 48% of training at full LR — a modest increase. This is a zero-size-risk schedule change (priority 1 in the research program) that continues the warmdown sweep to bracket the optimum around 600.

## Diff from previous best
+2 lines / -2 lines (vs current best)
