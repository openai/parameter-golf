# Experiment 8

**Date:** 2026-03-19T19:18:32.282655+00:00
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
**Propose time:** 101.9s
**Train time:** 968.9s
**Error:** timeout

## Change
Add post-quantization skip-weight scale calibration (Phase 6). After int8 quantization, encoder weight matrices are quantized which shifts the magnitude of hidden states flowing through the U-Net skip connections. This adds a global scalar multiplier on skip_weights and searches for the optimal value (coarse grid [0.8-1.2] + fine ±0.04 around best) using fast no-overlap eval. The buffer is initialized to 1.0 so training is unaffected; only the post-quantization calibration search modifies it. This is the #1 priority in the eval_time lane and targets a genuinely independent calibration axis from temperature (linear logit scaling) and softcap (nonlinear logit clipping).

## Diff from previous best
+24 lines / -1 lines (vs current best)
