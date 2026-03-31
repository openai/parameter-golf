# Experiment 8

**Date:** 2026-03-19T18:23:32.277243+00:00
**Lane/Stage:** storage/discovery
**Result:** REVERTED
**val_bpb:** 1.6805
**Artifact size:** 8,594,255 bytes
**Model params:** 17059912
**Last step:** 319
**Pre-quant val_bpb:** 1.6408
**Quantization gap:** 0.0397
**Eval time:** 11006 ms
**Peak memory:** 10039 MiB
**Gate reason:** no_storage_improvement
**Propose time:** 207.9s
**Train time:** 313.4s

## Change
Add straight-through int8 quantization-aware training (QAT) to CastedLinear forward pass. During training, weights are simulated through per-row int8 quantize→dequantize with a straight-through estimator so gradients flow through original weights. This teaches the model to learn weights robust to int8 rounding, directly reducing the quantization gap (currently 0.0287). The QAT uses per-row amax scaling matching the export scheme. Compute overhead is negligible (~0.01%) since ops are element-wise and tiny compared to matmuls. Combined with the already-present Muon weight decay (0.02), this provides both magnitude regularization and quantization-noise robustness.

## Diff from previous best
+16 lines / -3 lines (vs current best)
