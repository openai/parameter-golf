# Experiment 9

**Date:** 2026-03-19T19:36:23.550602+00:00
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
**Train time:** 1059.5s
**Error:** timeout

## Change
Add post-quantization skip_weight scale calibration (Phase 6). After int8 quantization shifts the relative magnitudes of encoder/decoder layer outputs, the optimal balance between U-Net skip connections and the main residual stream changes. A new `eval_skip_scale` buffer (initialized to 1.0, no effect during training) multiplies all skip_weights during forward. Post-quantization, a coarse grid [0.7-1.3] + fine ±0.04 search finds the optimal scale. This is an independent calibration axis from temperature (linear logit scaling) and softcap (nonlinear logit clipping) — it targets activation-path distortion rather than output-logit distortion. Uses ~19 fast no-overlap evals.

## Diff from previous best
+26 lines / -1 lines (vs current best)
