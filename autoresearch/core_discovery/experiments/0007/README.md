# Experiment 7

**Date:** 2026-03-19T18:23:31.116975+00:00
**Lane/Stage:** core/discovery
**Result:** KEPT
**val_bpb:** 1.5247
**Artifact size:** 12,483,023 bytes
**Model params:** 18897488
**Last step:** 346
**Pre-quant val_bpb:** 1.5208
**Quantization gap:** 0.0039
**Eval time:** 12307 ms
**Peak memory:** 11397 MiB
**Gate reason:** improved_val_bpb (1.5574078 -> 1.5247)
**Propose time:** 88.3s
**Train time:** 253.7s

## Change
Reduce warmdown_iters from 1200 to 800 to give the model ~400 more steps at full learning rate on the short-horizon proxy. With ~4186 steps in 180s, 1200 warmdown steps means ~29% of training is in cooldown; 800 steps reduces this to ~19%, allowing more productive learning time before decay. This should improve convergence without sacrificing end-of-training stability, since the warmdown was designed for 20,000-iteration runs and is disproportionately long for short-horizon training. Zero impact on step time or artifact size.

## Diff from previous best
Identical to current best
