# Experiment 17

**Date:** 2026-03-19T19:23:11.480808+00:00
**Lane/Stage:** storage/discovery
**Result:** REVERTED
**val_bpb:** 1.6658
**Artifact size:** 8,761,726 bytes
**Model params:** 17059912
**Last step:** 327
**Pre-quant val_bpb:** 1.6324
**Quantization gap:** 0.0334
**Eval time:** 11030 ms
**Peak memory:** 10240 MiB
**Gate reason:** no_storage_improvement
**Propose time:** 0.0s
**Train time:** 281.9s

## Change
Add per-row mean centering to int8 quantization. Before quantizing each row, subtract its mean and quantize the centered (zero-mean) residual. The per-row mean is stored as fp16 and added back during dequantization. This converts symmetric quantization into asymmetric, better utilizing the full [-127, 127] int8 range for rows whose weight distribution has shifted away from zero during training. The overhead is ~56KB of fp16 means (negligible vs 8.77MB artifact). Training is completely unchanged — only the export-time quantization/dequantization paths are modified.

## Diff from previous best
+58 lines / -27 lines (vs current best)
