# Experiment 100

**Date:** 2026-03-19T15:28:13.297917+00:00
**Result:** REVERTED
**val_bpb:** 1.7083
**Artifact size:** 13,865,490 bytes
**Propose time:** 228.2s
**Train time:** 356.8s

## Change
Replaced ReLU² MLP with SwiGLU (gated SiLU) MLP at matched parameter count. SwiGLU uses gate+up+proj projections with hidden=2/3×mlp_mult×dim=1024, giving exactly the same 1,572,864 params/layer as the original 2-matrix ReLU² with hidden=1536. SwiGLU's gating mechanism is strictly more expressive than ReLU² and has been shown to outperform it in LLaMA/PaLM-scale experiments. Unlike experiment #1 which merely swapped the activation function (SiLU vs ReLU²), SwiGLU adds a learned gating mechanism that selectively modulates information flow, which is a fundamentally different architectural change.

## Diff from previous best
+7 lines / -7 lines (vs current best)
