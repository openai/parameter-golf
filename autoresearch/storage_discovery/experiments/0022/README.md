# Experiment 22

**Date:** 2026-03-19T20:16:10.174634+00:00
**Lane/Stage:** storage/discovery
**Result:** REVERTED
**val_bpb:** 1.6688
**Artifact size:** 8,701,919 bytes
**Model params:** 17059912
**Last step:** 326
**Pre-quant val_bpb:** 1.6331
**Quantization gap:** 0.0357
**Eval time:** 11010 ms
**Peak memory:** 10240 MiB
**Gate reason:** no_storage_improvement
**Propose time:** 0.0s
**Train time:** 274.2s

## Change
Three-pass coarse-to-fine MSE-optimal int8 quantization with wider initial range [0.30, 1.0] (from [0.50, 1.0]). Adds a third refinement pass for ~2.5x finer resolution (~0.001 vs ~0.0025) at the same total candidate count (45 vs 42). The wider range lets the search find optimal scales for outlier-heavy rows that need aggressive clipping below the previous 0.475 minimum. Training is completely unchanged — only the export-time quantization search is refined.

## Diff from previous best
+14 lines / -8 lines (vs current best)
