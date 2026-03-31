# Experiment 13

**Date:** 2026-03-19T18:52:47.822857+00:00
**Lane/Stage:** storage/discovery
**Result:** REVERTED
**val_bpb:** 1.6815
**Artifact size:** 8,648,376 bytes
**Model params:** 17059912
**Last step:** 326
**Pre-quant val_bpb:** 1.6314
**Quantization gap:** 0.0501
**Eval time:** 11007 ms
**Peak memory:** 10239 MiB
**Gate reason:** no_storage_improvement
**Propose time:** 0.0s
**Train time:** 250.2s

## Change
Lower INT8_CLIP_PERCENTILE from 99.5 to 99.0 to clip more weight outliers during per-row int8 quantization. Experiment #7 showed that going from 99.99984→99.5 improved post-quantization BPB by providing tighter per-row scales. This continues that direction: clipping the top 1% of absolute values per row (instead of 0.5%) gives even tighter scales and better int8 precision for the bulk of the weights. For a 512-element row, this clips ~5 extreme values instead of ~2-3, a modest increase that should further reduce quantization error across the remaining 99% of weights without materially hurting model quality.

## Diff from previous best
+2 lines / -1 lines (vs current best)
