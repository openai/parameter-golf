# CROWN-Q + Full GPTQ + SWA/EMA Blend

## Summary

- **CROWN-Q**: Curvature-weighted quantization variance penalty applied during warmdown. Encourages weights to settle in flat minima where int6 quantization causes less damage. Penalty: `lambda * sum(h_j * delta_j^2 / 12)` where `h_j = w^2` (curvature proxy) and `delta_j = row_max / 15` (quantization step size).
- **Full Cholesky GPTQ**: Hessian-aware quantization with act-order column permutation, block_size=128, 256-sample calibration from training data. All within 585s training budget.
- **SWA/EMA 50/50 blend**: Stochastic Weight Averaging (every 50 steps during warmdown) blended 50/50 with EMA (decay=0.997).
- **Architecture**: 11L, 512d, GQA 8H/4KV, MLP 3x LeakyReLU(0.5)^2, XSA on all 11 layers, VRL, BigramHash 3072, partial RoPE 16/64.
- **Eval**: Sliding window with stride=64. No test-time training.

## Configuration

```bash
# Training (585s wallclock, includes GPTQ calibration)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Key env vars (all defaults in code):
# CROWNQ_LAMBDA=0.01        — CROWN-Q penalty weight
# CROWNQ_WARMDOWN_ONLY=1    — only apply during warmdown
# LATE_QAT_THRESHOLD=0.15   — QAT activation point
# MAX_WALLCLOCK_SECONDS=585  — training budget
# WARMDOWN_ITERS=4000        — warmdown length
```

## Results

| Seed | Steps | Post-EMA BPB | Sliding BPB | Artifact |
|------|-------|-------------|-------------|----------|
| 1337 | 6613  | 1.1387      | **1.1189**  | 15,945,134 |
| 42   | 6612  | 1.1382      | **1.1189**  | 15,947,742 |
| 7    | 6612  | 1.1378      | **1.1179**  | 15,938,790 |
| **Mean** | | 1.1382 | **1.1186** | |
| **Std** | | | 0.0006 | |

- Step speed: 87ms/step (FA3 Hopper)
- Quant gap (roundtrip): ~0.004 BPB
- Sliding window eval time: ~75s
- Training time: 585s (under 600s budget)

## What is CROWN-Q?

CROWN-Q (Curvature-Regularized Optimization for Weight Noise Quantization) adds a training-time penalty that makes weights more robust to quantization noise:

1. For each weight matrix, compute the per-row quantization step size `delta = row_max / 15` (int6 range [-15, 15])
2. Compute quantization variance `delta^2 / 12` (uniform rounding noise)
3. Weight by curvature proxy `h = w^2` (large weights in high-curvature directions)
4. Penalty: `lambda * sum(h * quant_var)` encourages the optimizer to reduce weights in directions where quantization noise is most damaging

Applied only during warmdown when QAT is active. Zero eval-time cost.

## Included Files

- `train_gpt.py` — self-contained training script
- `submission.json` — submission metadata
- `README.md` — this file
- `train_seed1337.log` — seed 1337 training log
- `train_seed42.log` — seed 42 training log
- `train_seed7.log` — seed 7 training log
