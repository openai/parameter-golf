# Experiment 10

**Date:** 2026-03-19T19:54:03.692617+00:00
**Lane/Stage:** eval_time/discovery
**Result:** REVERTED
**val_bpb:** N/A (failed)
**Artifact size:** N/A bytes
**Model params:** N/A
**Last step:** N/A
**Pre-quant val_bpb:** N/A
**Quantization gap:** N/A
**Eval time:** N/A ms
**Peak memory:** N/A MiB
**Gate reason:** timeout
**Propose time:** 0.0s
**Train time:** 910.2s
**Error:** timeout

## Change
Add post-quantization skip-weight scale calibration (Phase 6) by directly scaling the skip_weights parameter data in-place after dequantization, avoiding forward-pass changes or recompilation that caused the previous attempt (#8) to timeout. After int8 quantization, encoder weight matrices shift magnitude, so hidden states flowing through U-Net skip connections change scale. Searches a 5-point coarse grid [0.90-1.10] + 4-point fine grid around best for optimal global multiplier on skip_weights, using fast no-overlap eval (9 total passes ≈ 9-18s additional eval time).

## Diff from previous best
+24 lines / -0 lines (vs current best)
