# Full GPTQ + LeakyReLU² + Parallel Muon

**val_bpb: 1.1171** (3-seed mean, std 0.0003) | **~15.95 MB** | 8×H100 SXM, 600s | No TTT

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-quant bpb | **Post-GPTQ sliding bpb** | Artifact |
|------|----------|-------|---------------|--------------------------|----------|
| 2025 | 83.4ms | 7,182 | 1.1385 | **1.1167** | 15,901,230 |
| 1337 | 83.3ms | 7,189 | 1.1388 | **1.1171** | 15,962,990 |
| 2024 | 83.3ms | 7,201 | 1.1386 | **1.1173** | 15,994,746 |
| **Mean** | **83.3ms** | **7,191** | **1.1386** | **1.1171 (std 0.0003)** | |

GPTQ improves post-quantization BPB by **0.0216** vs pre-quantization (1.1386 → 1.1170). Standard GPTQ-lite gives only 1.1218 from the same pre-quant model — Full GPTQ is 0.0048 better.

## Key Innovation: Full Hessian GPTQ

Standard GPTQ-lite searches for the best per-row clip percentile — a greedy row-wise optimization. Full GPTQ uses second-order information (the Hessian H = X^T X) to compensate for quantization error across columns:

1. **Hessian collection**: 256 calibration batches through a non-banked model replica, accumulating H = X^T X per linear layer via forward hooks
2. **Column reordering (actorder)**: Quantize columns in order of descending Hessian diagonal (most important first)
3. **Cholesky error compensation**: For each column block, propagate quantization error to remaining columns using H^{-1}, minimizing total reconstruction loss
4. **Per-row scale search**: Same 5-percentile search as GPTQ-lite, but applied within the Cholesky framework

Based on IST-DASLab/gptq (ICLR 2023). Adapted for banked weights by unbanking to a temporary non-banked model for Hessian collection.

## Training Architecture

PR #414 stack with Parameter Banking + Parallel Muon ([PR #399](https://github.com/openai/parameter-golf/pull/399)):

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with **LeakyReLU(0.5)²** |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | **Full Hessian GPTQ int6** + lzma |
| Optimizer | Parameter Banking + Parallel Muon |

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Eval Timing

| Phase | Time |
|-------|------|
| Training | 600s |
| Hessian collection (256 batches) | ~25s |
| GPTQ quantization | ~60s |
| Sliding window eval (stride=64) | ~100s |
| **Total eval** | **~185s (< 10 min)** |

No TTT needed — Full GPTQ alone beats all prior TTT-based submissions.

## Credits

- **Full GPTQ**: PR #569 by @abaybektursun (Hessian-aware quantization implementation)
- **LeakyReLU²**: PR #493, PR #518
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
- **GPTQ algorithm**: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (ICLR 2023)
