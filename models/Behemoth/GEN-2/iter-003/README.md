# Behemoth GEN-2 iter-003: Meta-Preconditioner (Learned Gradient Geometry)

- **Goal**: Force the Macro Teacher to learn a dynamic preconditioner inside the forward pass, steering gradients during backprop via learned coordinate transformations.
- **Parent**: `models/Behemoth/GEN-2/iter-001` (Orthogonal Temporal Router baseline, 1.4880 BPB)

## Core Thesis

Standard deep learning assumes a fixed Euclidean geometry for the loss landscape. By inserting a **learned, input-dependent coordinate transformation** into the forward pass, the model approximates its own optimal gradient geometry — akin to Natural Gradient Descent, but entirely learned and dynamic.

The key insight: in PyTorch, the backward pass is strictly dictated by the chain rule on the forward graph. If the Macro Teacher predicts a transformation matrix $T$ applied as $y = T \cdot x$, then during backprop:

$$\nabla_x \mathcal{L} = T^T \cdot \nabla_y \mathcal{L}$$

The Teacher's transpose $T^T$ **literally rotates and scales the gradient vectors** before they flow backward into the encoder blocks. The network learns how to steer its own gradients.

## Variants

This iteration explores two lightweight proxies for a full coordinate transformation, then combines them:

### Option 1: FiLM — Dynamic Diagonal Preconditioning (`train_gpt.py`)

The Teacher predicts a per-dimension scaling vector $\gamma$ via a linear head + softplus activation. The Student's cross-attention context is element-wise scaled:

$$x' = \gamma \odot x$$

- **Gradient effect**: Each dimension's gradient is dynamically amplified or throttled based on what the Teacher considers important for that temporal interval.
- **Cost**: $O(D)$ — essentially free.
- **Implementation**: `macro_film_scale` linear layer (D→D with bias), initialized to near-identity (weights=0, bias=0 → softplus(0) ≈ 0.693).

### Option 2: Low-Rank Orthogonal Shift (`train_gpt_3.1.py`)

The Teacher predicts two narrow matrices $U \in \mathbb{R}^{D \times R}$ and $V \in \mathbb{R}^{R \times D}$ forming a low-rank update to the identity:

$$x' = (I + UV^T) x$$

- **Gradient effect**: Backprop multiplies gradients by $(I + VU^T)$. The model learns to identify an $R$-dimensional subspace where gradients need massive redirection, leaving the rest untouched.
- **Cost**: $O(R \cdot D)$ with $R=16$ — very efficient.
- **Implementation**: `macro_lowrank_U` (R→D) and `macro_lowrank_V` (D→R), small-norm init (std=0.01).

### Combined: FiLM + Low-Rank R=48 (`fury/train_gpt.py`)

Stacks both mechanisms for full meta-preconditioning:

$$x' = \gamma \odot (I + UV^T) x$$

- **Diagonal scaling** (FiLM) handles per-dimension importance weighting.
- **Subspace rotation** (low-rank) handles gradient redirection in a learned subspace.
- **Rank increased** from 16 to 48 for greater expressivity (~49K additional params).

## Architecture Diagram

```text
[ Encoder Blocks ]
        |
        v
+-------------------+
| Interval Summary  |  <-- Last token per macro interval
+--------+----------+
         |
   Teacher + Student Distillation (MSE)
         |
         v
+-------------------+
| Cross-Attention   |  <-- Tokens attend to past summaries
+--------+----------+
         |
         v
+-------------------+
| RMSNorm(context)  |  <-- Variance stabilization
+--------+----------+
         |
         v
+-------------------+
| FiLM Scale (γ)    |  <-- Option 1: diagonal preconditioning
| softplus(linear)  |      ∇_x L = γ · ∇_y L
+--------+----------+
         |
         v
+-------------------+
| Gated Injection   |  <-- sigmoid(gate) * scaled_context
+--------+----------+
         |
         v
+-------------------+
| Low-Rank Shift    |  <-- Option 2: x = x + U(V^T x)
| (I + UV^T)x       |      ∇_x L = (I + VU^T) ∇_y L
+--------+----------+
         |
         v
[ Decoder Blocks ]
```

## Results

| Variant | File | val_bpb | Artifact Size | Notes |
|---------|------|---------|---------------|-------|
| iter-001 baseline | — | 1.4880 | 14.56 MB | No meta-preconditioner |
| Option 1: FiLM | `train_gpt.py` | **1.4571** | 14.79 MB | FiLM weights were orphaned from optimizer (never trained!) — improvement came from architectural changes alone |
| Option 2: Low-Rank R=16 | `train_gpt_3.1.py` | 1.4661 | 14.58 MB | Low-rank weights initially orphaned, fixed mid-run |
| Combined: FiLM + Low-Rank R=48 | `fury/train_gpt.py` | *running* | — | Clean run with all weights properly in optimizer groups |

**Key finding**: Option 1 (FiLM) beat the baseline by **0.031 BPB** (1.4571 vs 1.4880) despite the FiLM scale weights never actually being trained due to an optimizer bug. This suggests the architectural pathway itself (cross-attention → RMSNorm → gated injection) was the primary driver, and properly training the meta-preconditioner should yield further gains.

## Bug Fixes During Iteration

### Orphaned Parameter Bug (Critical)

Both `macro_film_scale` (Option 1) and `macro_lowrank_U`/`V` (Option 2) were initialized but **never included in any optimizer group**. The parameters existed in the model but received zero gradient updates during training.

**Fix for Option 1** (`train_gpt.py`):
- Added `macro_film_scale.weight` to `collect_matrix_params()` (Muon optimizer, 2D matrix param)
- Added `macro_film_scale.bias` to `scalar_params` (AdamW optimizer, 1D param)

**Fix for Option 2** (`train_gpt_3.1.py`):
- Added `macro_lowrank_U.weight` and `macro_lowrank_V.weight` to `collect_matrix_params()` (Muon optimizer)

The `fury/` variant was created after these fixes, so all parameters are properly optimized from step 0.

## Hyperparameters

```
model_params: ~33.4M
dim: 512
macro_interval: 16
macro_xattn_dim: 128
macro_distill_weight: 0.150
macro_lowrank_r: 48 (fury), 16 (3.1)
attention_mode: gqa (8 heads, 4 kv heads)
block_schedule: parallel
obf_weight: 0.003 (start step 500)
adaptive_depth: True
u_net_skips: True
macro_pyramid: True
```

## File Manifest

```
iter-003/
├── README.md              ← This file
├── train_gpt.py           ← Option 1: FiLM dynamic diagonal preconditioning
├── train_gpt_3.1.py       ← Option 2: Low-rank orthogonal shift (R=16)
├── logs.txt               ← Training log for Option 1 (val_bpb: 1.4571)
├── logs_3.1.txt           ← Training log for Option 2 (val_bpb: 1.4661)
└── fury/
    ├── train_gpt.py       ← Combined: FiLM + Low-Rank R=48
    └── logs.txt           ← Training log (in progress)
```

## Next Steps

1. Analyze `fury/` results — does the combined FiLM + low-rank R=48 with properly trained weights beat the FiLM-only 1.4571?
2. If combined wins: sweep rank R ∈ {32, 48, 64, 96} to find optimal expressivity vs. cost tradeoff.
3. If FiLM-only still dominates: the diagonal approximation may be sufficient and the subspace rotation adds noise. Increase FiLM capacity instead (e.g., 2-layer MLP predicting γ).
4. Consider orthogonality constraint on U, V to prevent degenerate low-rank subspaces.

## References

- **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
- **Natural Gradient**: Amari, "Natural Gradient Works Efficiently in Learning" (1998)
- **Low-Rank Adaptation**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **Universal Transformer**: Dehghani et al. (2018) — depth recurrence
- **ALBERT**: Lan et al. (2019) — parameter sharing
