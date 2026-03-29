# Depth-Recurrent UT + Rank-1 LoRA Per-Iteration Adaptation — val_bpb 1.3342

**val_bpb = 1.3342** (1 seed, additional seeds pending H100 access) | **11.39 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.7.1)

| Seed | step_avg | steps | pre_quant_bpb | roundtrip_bpb | sliding_bpb | Artifact |
|------|----------|-------|---------------|---------------|-------------|----------|
| 1337 | 125ms | 4,769 | 1.343 | 1.359 | **1.334** | 11,385,022 |
| 42 | — | — | — | — | — | pending |
| 2025 | — | — | — | — | — | pending |

> Additional seeds pending H100 access.

## Key Innovation: Rank-1 LoRA for Stable Per-Iteration Adaptation

This submission introduces **rank-1 LoRA** as the first stable mechanism for per-iteration weight adaptation in depth-recurrent transformers. Each shared block's Q, V, MLP-up, and MLP-down matrices get a unique rank-1 modification at each loop iteration:

```python
# Rank-1 delta = outer product of two learned vectors
# b: (out_dim,), a: (in_dim,) — both on AdamW, NOT Muon
delta_W = b.unsqueeze(1) * a.unsqueeze(0)  # rank-1 matrix
W_effective = W_shared + delta_W            # unique per iteration
```

Total rank-1 params: ~9K (negligible — 0.04% of model). The vectors are stored on **AdamW** (not Muon), which is critical for stability.

### Why Rank-8 LoRA Diverges (and Rank-1 Doesn't)

We conducted 8 training runs with rank-8 LoRA, systematically varying optimizer (Muon vs AdamW), learning rate (0.005-0.010), warmup strategy (0-2000 steps), and gradient scaling (1/num_loops). **All 8 diverged** between steps 1500-4000.

**Root cause: Muon's Newton-Schulz scale factor asymmetry.**

Muon applies `scale = sqrt(rows/cols)` to each parameter update. For rank-8 LoRA:

| Matrix | Shape | Muon scale |
|--------|-------|:---:|
| LoRA B (out x rank) | (576, 8) | sqrt(72) = **8.49x** |
| LoRA A (rank x in) | (8, 576) | max(1, 8/576)^0.5 = **1.0x** |

The B matrices receive updates **8.5x larger** than A matrices. This creates a positive feedback loop: B grows fast, which increases dL/dA (since dA = B^T @ dL/dW), which grows A, which makes B@A larger, accelerating divergence.

**Rank-1 fix**: With rank=1, the LoRA params are 1D **vectors** (not 2D matrices), so they go to AdamW instead of Muon. AdamW has no aspect-ratio scaling — problem eliminated.

| Attempt | Optimizer | LR | Fix | Result |
|---------|----------|:---:|-----|--------|
| v1 | Muon | 0.010 | None | Diverged step 1500 |
| v2 | Muon | 0.005 | Lower LR | Diverged step 1500 |
| v3 | AdamW (LoRA only) | 0.025 | Separate optimizer | Slow convergence, diverged step 3000 |
| v4 | Muon | 0.010 | Grad zero warmup + 1/3 scale | Grad clip bug: LoRA inflated global norm |
| v5 | Muon | 0.010 | Fixed clip ordering + warmup | Diverged step 3500 (1500 after unfreeze) |
| v6 | Muon (scale=1.0 override) | 0.010 | Override Muon scale for LoRA | Diverged step 4000 |
| v7 | AdamW | 0.010 | LoRA warmup 2000 steps | Partial — survived to end but noisy |
| **v8 (this)** | **AdamW, rank-1** | **0.010** | **Vectors, not matrices** | **Stable! 1.334 BPB** |

## Architecture

640d model that **cannot fit as a flat transformer** in 16 MB (would be 18.2 MB at INT6). Depth recurrence enables this width.

| Parameter | Value |
|-----------|-------|
| Structure | 1 prelude + 4 shared x 3 loops + 1 coda |
| Effective layers | 14 (from 6 unique blocks) |
| Model dim | 640 |
| Heads / KV heads | 10 / 5 (head_dim=64) |
| MLP multiplier | 3.0 (hidden=1920) |
| Activation | LeakyReLU(0.5) squared |
| Rank-1 LoRA | On Q, V, MLP-up, MLP-down per shared effective layer |
| Total rank-1 params | ~9K |
| Vocab | 1024 BPE, tied embeddings |

### Stability Techniques

- **Output-LN (Peri-LN)**: RMSNorm on attn/MLP output (not input) for shared blocks. Prevents magnitude information loss across loop iterations. (arXiv:2502.02732)
- **Birkhoff-constrained mixing**: `alpha = sigmoid(logit)` for residual mixing, guaranteeing spectral norm <= 1. Prevents signal blowup. (PR #855)
- **Capped timestep scaling**: Per-effective-layer scale vectors clamped to [-4, 4], stored as FP16 passthrough. Reduces quantization gap by 26-30%.
- **Noisy QAT**: INT6-calibrated noise on shared block weights during training.

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer (banks) | Muon (NS5, momentum 0.99) |
| Optimizer (rank-1 LoRA, scalars) | AdamW |
| Matrix LR | 0.010 |
| Grad clip norm | 0.3 |
| Weight decay | 0.04 |
| Batch tokens | 524,288 |
| EMA decay | 0.997 |

## Artifact

Only **11.39 MB** — leaves **4.61 MB free** for potential n-gram cache integration.

```
Shared block weights (INT6 GPTQ):  ~10.5 MB
Rank-1 LoRA vectors (FP16):       ~0.02 MB
Embedding + controls:              ~0.8 MB
Code:                              ~0.1 MB
Total:                             11.39 MB
```

## Credits

- PR #855 (@aazizyan) — Output-LN, Birkhoff mixing, timestep scaling (first viable 3-loop recurrence)
- PR #895 (@iverbovoy) — Progressive depth, loop embedding concept
- PR #363 (@evangelinehelsinki) — Noisy QAT for recurrence, negative results documentation
- arXiv:2502.02732 — Peri-LN normalization
- arXiv:2502.13181 — RingFormer level signals (inspiration for per-iteration adaptation)
