# Experiment 18

**Date:** 2026-03-19T19:42:19.661245+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5245
**Artifact size:** 12,302,044 bytes
**Model params:** 17342176
**Last step:** 345
**Pre-quant val_bpb:** 1.5225
**Quantization gap:** 0.0020
**Eval time:** 15291 ms
**Peak memory:** 12106 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5245)
**Propose time:** 71.7s
**Train time:** 264.2s

## Change
Reduce warmdown_iters from 600 to 550 to test if a slightly more aggressive warmdown schedule improves convergence on the 180s proxy. With ~377 steps at ~477ms/step, warmdown_iters=550 gives warmdown_ms=262s, so the starting LR is ~69% of base (vs ~63% at 600). The U-shaped curve (800→1.5247, 600→1.4967, 400→1.5174) suggests the optimum is near 600; this probes one small step toward higher initial LR to find the local minimum. Zero impact on step time, parameters, or artifact size.

## Diff from previous best
+2 lines / -2 lines (vs current best)
