# Experiment 103

**Date:** 2026-03-19T15:59:49.109067+00:00
**Result:** REVERTED
**val_bpb:** 1.5874
**Artifact size:** 15,400,923 bytes
**Model params:** 28863584
**Last step:** 321
**Pre-quant val_bpb:** 1.5699
**Quantization gap:** 0.0175
**Propose time:** 253.1s
**Train time:** 281.8s

## Change
Increased scalar_lr from 0.04 to 0.06 (+50%). The scalar parameters (attn_scale, mlp_scale, resid_mix, q_gain, skip_weights) control critical information routing and gating throughout the 12-layer network. Faster convergence of these gating parameters should allow the network to optimize its information flow earlier, making the Muon-trained matrix parameters more effective. This has zero impact on artifact size since scalar params are tiny and stored as passthrough in quantization.

## Diff from previous best
+1 lines / -1 lines (vs current best)
