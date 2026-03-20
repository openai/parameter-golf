# Complexity MoE + PID Dynamics

**Author:** Boris Peyriguere (Complexity-ML)
**Date:** 2026-03-20
**Score:** Pending (awaiting compute credits)

## Summary

Novel architecture combining innovations from the [Complexity Framework](https://github.com/Complexity-ML/complexity-framework):

1. **Token-Routed MoE** — 4 experts with deterministic routing (`token_id % 4`). Zero-overhead, perfectly balanced, no auxiliary loss needed. Mask-multiply pattern for `torch.compile(fullgraph=True)` compatibility.

2. **PID Dynamics (INL Ultra-Lite)** — Learnable equilibrium `mu` traverses all 9 layers. Fixed alpha/beta/gate with tight clamping (mu ∈ [0.5, 1.5], velocity ∈ [-3, 3]). Stabilises hidden state trajectories like a PID controller.

3. **SwiGLU Activation** — Replaces relu² from baseline. Gate+up fused projection per expert.

4. **Cosine Warm Restarts (SGDR)** — LR schedule with increasing cycle lengths (5k/10k/20k steps). Peak LR decays 0.7× each restart. Drives PPL lower than linear warmdown.

## Architecture

```
Input → Embedding → RMSNorm → [Block × 9] → Final Norm → LM Head

Block:
  ResidMix → Attention(GQA) → PID(h, v) → RMSNorm → TokenRoutedMLP(4 experts)
                                  ↕
                           mu, velocity traverse all layers
```

## Parameters

- **14.7M params** (same budget as baseline ~5.3M attention + ~9.4M MLP)
- Under 16MB after int8+zlib compression
- Config-driven via `config.json` for deterministic reproducibility

## Key Differences from Baseline

| Aspect | Baseline | This Submission |
|--------|----------|----------------|
| MLP | Dense relu² | Token-Routed 4× SwiGLU experts |
| Routing | N/A | `token_id % 4` (deterministic) |
| Dynamics | None | PID (mu across all layers) |
| LR Schedule | Linear warmdown | Cosine warm restarts (SGDR) |
| Activation | relu² | SwiGLU |

## How to Run

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
RUN_ID=complexity_v1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Status

Awaiting RunPod compute credits to produce training logs and final val_bpb score.
