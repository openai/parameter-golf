# Experiment 101

**Date:** 2026-03-19T15:37:58.350278+00:00
**Result:** REVERTED
**val_bpb:** 1.6350
**Artifact size:** 15,380,577 bytes
**Propose time:** 0.0s
**Train time:** 283.8s

## Change
Switched MLP from ReLU² (relu then square, 2 projections) to SwiGLU (gated activation with SiLU, 3 projections) and reduced mlp_mult from 3 to 2 to keep parameter count identical. SwiGLU uses a multiplicative gating mechanism (SiLU(xW_gate) * xW_up) that is fundamentally different from experiment #1's simple activation swap — the gating provides much richer feature interactions. SwiGLU is the standard MLP in LLaMA, Gemma, Mistral, and most modern LLMs because it consistently outperforms ungated MLPs at equivalent parameter budgets.

## Diff from previous best
+4 lines / -4 lines (vs current best)
