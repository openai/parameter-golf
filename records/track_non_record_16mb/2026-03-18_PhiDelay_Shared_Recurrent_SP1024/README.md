# PhiDelay-Inspired Shared-Recurrent Language Model

**Author:** Bahman Masarrat
**Contact:** bmasarrat@gmail.com
**Date:** 2026-03-18
**Track:** Non-record, unlimited compute, 16 MB artifact cap

## Overview

This submission replaces standard self-attention with a lightweight multi-head delay mixer (PhiDelayLayer) and introduces recurrent weight sharing to maximize model quality under strict artifact-size constraints. Three physical transformer blocks are applied four times each (effective depth 12), yielding a 16.8M-parameter model that compresses to a 6.84 MB int8+zlib artifact — well under the 16 MB cap.

The key finding is that training duration, not architecture complexity, was the dominant bottleneck. A simple, stable architecture with shared weights and sufficient training time produced consistently better results than more exotic approaches (e.g., ternary quantized training via BitLinear).

## Key Contributions

**1. Effective depth via recurrent weight sharing.**
Instead of 9 independently parameterized blocks, the model uses 3 physical blocks repeated 4 times. This reduces unique parameters by ~65% while preserving effective depth. Learned per-block recurrence gates (`x = x + gate * (block(x, x0) - x)`) stabilize the repeated application.

**2. Artifact-aware model scaling.**
Progressive scaling from dim=512 to dim=1024 was guided by artifact-size estimates at each step. The final model uses only 43% of the 16 MB budget, leaving substantial headroom for future improvements.

**3. Training as the primary bottleneck.**
The largest single improvement (+0.45 BPB) came from extending training from 18 to 40 steps — not from any architectural change. The final 60-step run pushed val_bpb from 3.29 to 3.23, confirming that the architecture had not saturated.

**4. Compression stability.**
The int8+zlib roundtrip introduces essentially zero degradation (val_bpb 3.2306 pre-quant vs 3.2306 post-quant), indicating that the learned weights are well-conditioned for post-training quantization.

## Final Results

| Metric | Value |
|---|---|
| val_loss | 5.5573 |
| val_bpb | **3.2306** |
| int8+zlib roundtrip val_bpb | **3.2306** |
| int8+zlib artifact size | **6.84 MB** |
| Total parameters | 16,789,558 |
| Physical blocks | 3 |
| Effective depth | 12 (3 blocks x 4 repeats) |
| Model dimension | 1024 |
| Training steps | 60 |
| Warmup steps | 10 |

### Progressive Scaling History

| Configuration | Params | Eff. Depth | Steps | val_bpb | Artifact |
|---|---|---|---|---|---|
| dim=512, r=3 | 4.5M | 9 | 12 | 4.3253 | 1.77 MB |
| dim=768, r=3 | 9.6M | 9 | 14 | 3.9735 | 3.41 MB |
| dim=1024, r=3 | 16.8M | 9 | 16 | 3.7627 | 5.51 MB |
| dim=1024, r=4 | 16.8M | 12 | 18 | 3.7362 | 5.68 MB |
| dim=1024, r=4 | 16.8M | 12 | 40 | 3.2887 | 6.67 MB |
| **dim=1024, r=4** | **16.8M** | **12** | **60** | **3.2306** | **6.84 MB** |

## Key Insights

1. **Training duration dominated.** The jump from 3.74 to 3.23 BPB came almost entirely from more training steps, not architecture changes. The model was undertrained at 18 steps and still improving at 60.

2. **Parameter reuse outperformed aggressive quantization.** A BitLinear (ternary weight) variant achieved 6.09 BPB — far worse than the shared-recurrent model at 3.23 BPB with standard CastedLinear weights. Recurrence was a more effective parameter-efficiency strategy than quantized training for this task.

3. **Effective depth scaling helped, but with diminishing returns.** Going from 3 to 4 repeats (depth 9 to 12) improved BPB by 0.03 at matched step count. The larger gain came from training longer at the higher depth.

4. **The final model uses only 43% of the artifact budget.** At 6.84 MB against a 16 MB cap, there is significant room for further scaling (larger dim, more blocks, or more repeats).

## Architecture Details

The PhiDelayLayer replaces standard causal self-attention with a dynamic sequence-shifting operator:

- **Per-head adaptive delay:** Each of 8 heads computes a content-dependent delay `tau` from input variance, using learned `alpha` and `beta` parameters.
- **Differentiable fractional shift:** Token representations are interpolated between neighboring positions based on `tau`, maintaining full differentiability.
- **Gated residual:** A learned scalar gate controls the contribution of the shifted output.
- **Linear projection:** A single `CastedLinear(dim, dim)` projection follows the shift.

Each Block applies: `resid_mix` blending with initial embeddings, PhiDelayLayer with scaled residual, then MLP (relu-squared) with scaled residual. Recurrence gates at the GPT level interpolate between the block output and the input at each application.

## Relation to PhiDelayNet

This work is conceptually inspired by PhiDelayNet ([DOI: 10.5281/zenodo.16790526](https://doi.org/10.5281/zenodo.16790526)), which explores adaptive delay operators for irregular time-series forecasting. In this submission, those ideas — specifically, content-dependent temporal shifting and learnable delay parameters — are translated into a parameter-efficient recurrent language-model setting under strict artifact constraints.

This submission does not claim continuity with the original PhiDelayNet benchmarks. The time-series forecasting setting and the language-modeling setting differ substantially in data, evaluation, and architecture requirements. The connection is conceptual: the core idea that adaptive, learnable delays can serve as a lightweight alternative to full attention mechanisms.

## Category

This is a **non-record submission**. It does not target the official under-10-minute 8xH100 record path. The submission ran locally on Apple Silicon (MLX) with a single training shard for rapid iteration.

The submission's value lies in the architecture and optimization insights: that shared-weight recurrence with a simple delay-based mixer can produce a well-compressed, stable language model far below the artifact cap, and that training duration is the dominant variable once the architecture is sound.

## Reproducibility

Training command (MLX, Apple Silicon):

```bash
RUN_ID=mlx_final_push \
ITERATIONS=60 \
TRAIN_BATCH_TOKENS=1024 \
GRAD_ACCUM_STEPS=1 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=2048 \
WARMUP_STEPS=10 \
TRAIN_SEQ_LEN=128 \
TRAIN_LOG_EVERY=1 \
MAX_VAL_BATCHES=10 \
python3 train_gpt_mlx.py
```

Required files: `train_gpt_mlx.py`, `bitlinear.py` (present but unused when `USE_BITLINEAR=False`).

Dataset: `fineweb10B_sp1024` (1 training shard). Tokenizer: `fineweb_1024_bpe.model` (SentencePiece, 1024 vocab).
