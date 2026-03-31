# Experiment 12

**Date:** 2026-03-19T18:47:35.947436+00:00
**Lane/Stage:** storage/discovery
**Result:** REVERTED
**val_bpb:** 1.6807
**Artifact size:** 8,666,092 bytes
**Model params:** 17059912
**Last step:** 331
**Pre-quant val_bpb:** 1.6253
**Quantization gap:** 0.0554
**Eval time:** 11006 ms
**Peak memory:** 10239 MiB
**Gate reason:** no_storage_improvement
**Propose time:** 0.0s
**Train time:** 249.8s

## Change
Lower INT8_CLIP_PERCENTILE from 99.5 to 99.0 to clip the top 1% of weight outliers per row during int8 quantization (up from 0.5%). This tightens per-row scales by ~10%, giving better int8 precision for the 99% of unclipped values. Experiment #7 showed that going from ~100% to 99.5% improved post-quantization BPB; this continues the same trend to find the optimal clip level. Pure export-time change — training is unaffected.

## Diff from previous best
+2 lines / -1 lines (vs current best)
