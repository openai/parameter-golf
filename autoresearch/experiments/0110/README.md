# Experiment 110

**Date:** 2026-03-19T16:54:24.744450+00:00
**Result:** REVERTED
**val_bpb:** 1.5731
**Artifact size:** 15,648,482 bytes
**Model params:** 28863584
**Last step:** 312
**Pre-quant val_bpb:** 1.5589
**Quantization gap:** 0.0142
**Propose time:** 469.7s
**Train time:** 280.9s

## Change
Increased qk_gain_init from 1.5 to 2.0. The q_gain parameter controls attention sharpness per head (applied after QK normalization). With the broken warmdown on the 1xH100 proxy (effective LR is only ~30% of base from step 0), learned parameters barely move from initialization over ~300 steps, so initialization has outsized importance. Sharper initial attention (higher q_gain) helps the model form more decisive attention patterns earlier, improving head specialization and gradient signal quality. With head_dim=64 and q_gain=2.0, initial attention logit std is ~2.0 (vs ~1.5 before), giving meaningfully sharper but not degenerate attention. The q_gain is per-head and learned, so gradient descent can correct if 2.0 is suboptimal. Zero impact on parameter count, artifact size, or step time.

## Diff from previous best
+1 lines / -1 lines (vs current best)
