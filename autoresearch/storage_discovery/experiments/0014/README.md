# Experiment 14

**Date:** 2026-03-19T18:56:58.239754+00:00
**Lane/Stage:** storage/discovery
**Result:** KEPT
**val_bpb:** 1.6561
**Artifact size:** 8,772,464 bytes
**Model params:** 17059912
**Last step:** 331
**Pre-quant val_bpb:** 1.6216
**Quantization gap:** 0.0345
**Eval time:** 11012 ms
**Peak memory:** 10239 MiB
**Gate reason:** improved_val_bpb (1.6563 -> 1.6561)
**Propose time:** 0.0s
**Train time:** 265.8s

## Change
Replace fixed-percentile (99.5%) int8 quantization with per-row MSE-optimal scale search. For each row of each 2D weight matrix, try 41 clip fractions from 0.80 to 1.00 of the row's max absolute value and select the scale that minimizes reconstruction MSE. This is strictly better than any single fixed percentile because it finds the best clip level independently per row. Only affects export-time quantization — training is completely unchanged, artifact format is identical (int8 weights + fp16 scales), and the search cost is negligible on CPU after training completes.

## Diff from previous best
Identical to current best
