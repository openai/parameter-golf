# Experiment 104

**Date:** 2026-03-19T16:08:44.080655+00:00
**Result:** REVERTED
**val_bpb:** 1.6159
**Artifact size:** 15,016,084 bytes
**Model params:** 28863584
**Last step:** 304
**Pre-quant val_bpb:** 1.5963
**Quantization gap:** 0.0196
**Propose time:** 0.0s
**Train time:** 281.3s

## Change
Increased scalar_lr (Adam LR for control parameters: attn_scale, mlp_scale, resid_mix, skip_weights, q_gain) from 0.04 to 0.06 (+50%). These ~28K parameters gate the residual stream and U-Net skip connections. With limited training time, faster adaptation of internal routing/gating should improve convergence, analogous to the successful matrix_lr increase (0.05→0.06) in experiment #102. Zero parameter count or step time change.

## Diff from previous best
+1 lines / -1 lines (vs current best)
