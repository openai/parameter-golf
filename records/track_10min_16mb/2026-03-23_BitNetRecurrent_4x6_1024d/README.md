# BitNet b1.58 Ternary QAT + Depth Recurrence

**val_bpb: TBD** (mean across 3 seeds — results pending H100 run)

## Core Idea

Two orthogonal, compounding innovations:

**1. Ternary QAT (BitNet b1.58 + LSQ)**
All weight matrices are quantized to `{-1, 0, +1}` during training via a
Straight-Through Estimator (STE). The quantization scale `alpha = exp(log_alpha)`
is a learnable scalar initialized from `median(|W|)` — robust to weight outliers
at small parameter counts (BitNet b1.58 Reloaded, arXiv:2407.09527).

The baseline int8+zlib export sees values tightly clustered at
`{-alpha, 0, +alpha}`, achieving ~1.7 bits/param vs ~8 bits/param for
standard float→int8. This gives **~4–5× more unique parameters in the same
16MB budget.**

**2. Depth Recurrence with Eval-Time Loop Scaling**
`K=4` unique transformer blocks are applied `N=6` times per forward pass during
training (24 effective layers). At evaluation, `M=12` loops are used — the same
weights run twice as many times for free. This is explicitly permitted by the FAQ
("evaluation at any sequence length... push the bounds of evaluation methods").

The `resid_mix` residual anchor (`x = mix[0]*x + mix[1]*x0`) stabilizes
representations across recurrence steps, preventing distribution shift when
increasing loop count at eval time.

## Key Techniques

| Technique | Source |
|-----------|--------|
| Ternary QAT + STE | BitNet b1.58 (arXiv:2402.17764) |
| LSQ learnable scale | BitNet b1.58 Reloaded (arXiv:2407.09527) |
| Depth recurrence | Geiping et al. (arXiv:2502.05171) |
| Eval-time loop scaling | Saunshi et al. (arXiv:2510.25741) |
| Muon + weight decay | modded-nanogpt |
| Sliding window eval (stride=64) | SlidingWindow submission (this leaderboard) |
| FP16 tied embedding export | SlidingWindow submission |
| RoPE, GQA, QK-Norm, ReLU² | modded-nanogpt / nanogpt speedrun |

## Model Config

| Param | Value |
|-------|-------|
| vocab_size | 1024 |
| model_dim | 1024 |
| num_heads | 16 |
| num_kv_heads | 4 |
| mlp_mult | 6 |
| num_unique_layers | 4 |
| num_recurrences (train) | 6 |
| num_eval_recurrences | 12 |
| effective_depth (train) | 24 |
| effective_depth (eval) | 48 |

## Compression Math

- 61.9M unique params × 1.7 bits/param (ternary + zlib) ≈ 13.1MB weights
- Embedding (fp16): ~1.0MB
- Code: ~32KB
- **Total artifact: ~14.2MB** (validated on M1 Max)

Compare to float16 baseline: ~15M params × 8 bits/param ≈ 14.7MB

Our artifact is similar size but represents **4× more unique parameters** and
**24 effective layers** vs 10 flat layers.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | TBD | TBD | TBD | TBD |
| 42   | TBD | TBD | TBD | TBD |
| 7    | TBD | TBD | TBD | TBD |
| **Mean** | **TBD** | **TBD** | | |

## Reproduce

```bash
# 3 seeds, 8×H100, ~10 min training + ~3 min sliding-window eval each
for SEED in 1337 42 7; do
  RUN_ID=bitnet_rec_seed${SEED} SEED=${SEED} \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee logs/seed${SEED}.txt
done
```

## Local Validation (M1 Max)

Approach was iteratively developed on Apple M1 Max 64GB using the MLX port
(`train_gpt_mlx.py` in repo root), confirming:
- Ternary compression: 61.9M params → 14.23MB artifact ✅
- LSQ scale: O(1) forward pass, 559ms/step on M1 Max ✅
- Eval loop scaling: 2× loops at eval confirmed working ✅
