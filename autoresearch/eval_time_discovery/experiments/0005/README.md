# Experiment 5

**Date:** 2026-03-19T18:33:34.207990+00:00
**Lane/Stage:** eval_time/discovery
**Result:** KEPT
**val_bpb:** 1.5611
**Artifact size:** 9,092,930 bytes
**Model params:** 17059912
**Last step:** 370
**Pre-quant val_bpb:** 1.5363
**Quantization gap:** 0.0248
**Eval time:** 44523 ms
**Peak memory:** 10239 MiB
**Gate reason:** improved_val_bpb (none -> 1.5611)
**Propose time:** 261.2s
**Train time:** 503.4s

## Change
Add two-phase temperature calibration — after the coarse 0.05-step grid search, do a fine 0.01-step search around the best temperature. Post-quantization logit scale shift makes precise temperature tuning especially valuable, and the ~8 extra fast (no-overlap) evals cost negligible time.

## Diff from previous best
Identical to current best
