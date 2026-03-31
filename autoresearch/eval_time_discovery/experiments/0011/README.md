# Experiment 11

**Date:** 2026-03-19T20:09:14.702037+00:00
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
**Train time:** 925.1s
**Error:** timeout

## Change
Add post-quantization skip-weight scale calibration (Phase 6) by directly scaling skip_weights.data in-place instead of modifying the forward function. After int8 quantization shifts encoder weight magnitudes, the optimal U-Net skip connection balance changes. A coarse grid [0.85-1.15] + fine ±0.03 search finds the optimal scale. This avoids the torch.compile recompilation that caused timeout in experiments #8 and #9, and targets activation-path distortion (independent from temperature/softcap logit calibration).

## Diff from previous best
+24 lines / -0 lines (vs current best)
