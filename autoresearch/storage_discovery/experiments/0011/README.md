# Experiment 11

**Date:** 2026-03-19T18:40:35.746347+00:00
**Lane/Stage:** storage/discovery
**Result:** REVERTED
**val_bpb:** 1.6678
**Artifact size:** 8,704,065 bytes
**Model params:** 17059912
**Last step:** 328
**Pre-quant val_bpb:** 1.6292
**Quantization gap:** 0.0386
**Eval time:** 11005 ms
**Peak memory:** 10239 MiB
**Gate reason:** no_storage_improvement
**Propose time:** 0.0s
**Train time:** 249.3s

## Change
Replace per-row int8 quantization with per-group quantization (group_size=128) for 2D weight matrices. Instead of one scale per row (output channel), each group of 128 elements gets its own fp16 scale factor. This gives 4x finer granularity for 512-col matrices and 8x for 1024-col matrices, dramatically reducing quantization error because local weight ranges no longer get dominated by distant outliers in the same row. The extra scale overhead is ~200KB (well within the 16MB budget), and training is completely unchanged — only the post-training export/dequant paths are modified.

## Diff from previous best
+27 lines / -4 lines (vs current best)
