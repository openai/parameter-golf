# Kreïn-Space Geometric Attention

**Score:** TBD (requesting H100 evaluation)
**Author:** Richie (Knox Research)
**Date:** 2026-04-05

## Summary

Replaces softmax attention with distance-based scoring in an indefinite inner product (Kreïn) space, using raw sigmoid gating instead of normalization.

## Architecture

**One-sided Kreïn geometric attention:**
- Each token gets a *center* (position in space) and *signed metric tensor* (ruler)
- Positive metric dimensions = spacelike, negative = timelike (learned during training)
- Score(i→j) = weighted distance using token i's metric: `−Σ_d m_i_d · (c_j_d − c_i_d)²`
- Raw sigmoid gating — no softmax, no normalization (three independent papers identify softmax normalization as root cause of depth waste)
- Asymmetric by construction — each token defines its own notion of "nearby"

**Key config changes vs baseline:**
| | Baseline | Geometric |
|---|---|---|
| Attention | Dot-product + softmax | Kreïn distance + sigmoid |
| Layers | 9 | 16 |
| Dim | 512 | 256 |
| Heads | 8 | 16 |
| MLP | 2x | 4x |
| Position encoding | RoPE | Implicit in learned distances |
| Normalization | Softmax (relative) | None (absolute) |
| Params | ~15.9M | ~11.3M |

**Why deeper + narrower:** Vanilla transformers exhibit "cornerstone layer pathology" — L0-L1 do all the work, later layers are wasted (confirmed at scales up to 66B). Geometric attention distributes computation across all layers (causal ablation: max 7.8x vs vanilla's 100x at L0). More layers = more advantage.

**Why bigger MLP:** With smaller residual dim, the MLP expansion provides the per-layer compute capacity. The geometric attention already compresses the residual into low-dimensional subspaces (participation ratio 1.6 vs 33.1 for vanilla), so a wide residual is wasted.

## Prior work

- Architecture published on Zenodo: https://zenodo.org/records/19198177
- 30M geometric model achieves 58 PPL vs 125 PPL for vanilla at same params (2.15x efficiency)
- Causal ablation study shows geometric attention eliminates cornerstone layer pathology
- Three independent papers (Nakanishi 2604.01178, Zhang et al. 2604.01193, + our work) converge on softmax normalization as root cause of transformer depth waste

## Changes from baseline `train_gpt.py`

1. Replaced `CausalSelfAttention` with one-sided Kreïn geometric attention
2. Changed model shape: 16L d=256 H=16 KV=8 MLP=4x
3. Removed RoPE (not needed — positions encoded in learned distances)
4. Shorter warmdown (800 vs 1200) — geometric models benefit from longer stable LR phase
5. All other infrastructure unchanged (Muon optimizer, quantization, UNet skips, evaluation)
