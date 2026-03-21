# Experiment 19

**Date:** 2026-03-19T19:37:53.941282+00:00
**Lane/Stage:** storage/discovery
**Result:** KEPT
**val_bpb:** 1.6496
**Artifact size:** 8,769,192 bytes
**Model params:** 17059912
**Last step:** 338
**Pre-quant val_bpb:** 1.6157
**Quantization gap:** 0.0339
**Eval time:** 11005 ms
**Peak memory:** 10240 MiB
**Gate reason:** improved_val_bpb (1.6510 -> 1.6496)
**Propose time:** 237.9s
**Train time:** 281.6s

## Change
Two-pass coarse-to-fine MSE-optimal int8 quantization search. Replaces the single-pass 41-candidate grid over [0.80,1.0] with a coarse pass (21 candidates over [0.50,1.0]) followed by a per-row fine pass (21 candidates around each row's coarse optimum). This gives 2x finer resolution (step 0.0025 vs 0.005) near the optimum and allows more aggressive clipping for outlier-heavy rows, at the same total iteration count (42 vs 41). Training is completely unchanged — only the export-time quantization search is refined.

## Diff from previous best
Identical to current best
