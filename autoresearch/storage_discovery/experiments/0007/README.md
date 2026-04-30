# Experiment 7

**Date:** 2026-03-19T18:17:16.455024+00:00
**Lane/Stage:** storage/discovery
**Result:** KEPT
**val_bpb:** 1.6563
**Artifact size:** 8,866,524 bytes
**Model params:** 17059912
**Last step:** 323
**Pre-quant val_bpb:** 1.6276
**Quantization gap:** 0.0287
**Eval time:** 11009 ms
**Peak memory:** 10239 MiB
**Gate reason:** first_successful_storage_result
**Propose time:** 39.6s
**Train time:** 293.8s

## Change
Lower INT8_CLIP_PERCENTILE from 99.99984 to 99.5 to clip more weight outliers during quantization. This trades accuracy on the top 0.5% most extreme weights for better int8 precision across the remaining 99.5%, which should reduce post-quantization BPB because the quantization scale per row will be tighter and waste fewer levels on rare outlier values.

## Diff from previous best
Identical to current best
