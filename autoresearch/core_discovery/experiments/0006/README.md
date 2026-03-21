# Experiment 6

**Date:** 2026-03-19T18:15:05.959786+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5801
**Artifact size:** 10,560,262 bytes
**Model params:** 18897488
**Last step:** 344
**Pre-quant val_bpb:** 1.5653
**Quantization gap:** 0.0148
**Eval time:** 12253 ms
**Peak memory:** 11398 MiB
**Gate reason:** no_val_bpb_improvement (best=1.5574, got=1.5801)
**Propose time:** 0.0s
**Train time:** 292.7s

## Change
Increase scalar_lr from 0.04 to 0.06 to improve convergence speed of scalar/control parameters (attn_scale, mlp_scale, resid_mix, skip_weights, q_gain). This mirrors the successful matrix_lr increase from 0.04→0.06 in experiment #3, applying the same learning rate boost to the Adam-optimized 1D parameters that control layer scaling and residual mixing, which should help them adapt faster in the short-horizon proxy.

## Diff from previous best
+2 lines / -2 lines (vs current best)
