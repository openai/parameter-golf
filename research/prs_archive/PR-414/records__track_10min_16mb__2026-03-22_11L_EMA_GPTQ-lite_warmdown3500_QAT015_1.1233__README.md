## Record: 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 (val_bpb: 1.1233)

**val_bpb: 1.1233** (sliding window stride=64, 3-seed mean) | **15.55 MB** (mean) | 8xH100 SXM, 600s

### Key Innovations Over PR #374

Two novel post-training optimizations plus training hyperparameter tuning on top of PR #374's architecture:

| Change | PR #374 | This | Impact |
|--------|---------|------|--------|
| **GPTQ-lite** | Fixed clip (row max) | 5 clip percentiles per row, pick min MSE | -0.0006 BPB (zero training cost) |
| **EMA** | None (Tight SWA only) | EMA decay=0.997 every step | -0.0006 BPB (smoother averaging) |
| **Warmdown** | 3000 | 3500 | -0.0002 BPB |
| **Late QAT threshold** | 0.1 | 0.15 | -0.0001 BPB (earlier fake quant, smaller quant gap) |
| **Total** | **1.1246** | **1.1233** | **-0.0013 BPB** |

### GPTQ-lite: Per-Layer Optimal Clip Percentile Search

Instead of using the row maximum for int6 quantization scale, we try 5 clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) per weight matrix row and pick the one minimizing reconstruction MSE. This is applied during post-training quantization with zero training cost.

### EMA Weight Averaging

Exponential moving average (decay=0.997) maintained every training step, applied before quantization. Stacks with Tight SWA — EMA provides continuous smoothing while SWA captures discrete checkpoints during warmdown.

### Results (3 seeds, 8xH100 SXM)

| Seed | Steps | val_loss | Sliding BPB (s64) | Artifact |
|------|-------|----------|-------------------|----------|
| **1337** | 7101 | 1.8958 | **1.1228** | 15.56 MB |
| 42 | ~7100 | 1.8972 | 1.1236 | 15.54 MB |
| 2024 | ~7100 | 1.8971 | 1.1236 | 15.59 MB |

**Mean: 1.1233 | Std: 0.0005** | Submitted: seed 1337 (best)

### Architecture (from PR #374)

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), relu-squared activation
- U-Net skip connections (5 encoder, 6 decoder)
- Efficient Partial XSA on last 4 layers (GQA-aware, zero-alloc)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10) with per-layer learned scales
- SmearGate + BigramHash (2048 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training

- FlashAttention 3 (Hopper-optimized)
- Muon optimizer (matrices): lr=0.025, momentum=0.99 (warmup 0.92->0.99 over 1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3500 iterations (wallclock-based)
- **EMA**: decay=0.997, every step
- **Tight SWA**: every 50 steps when scale<0.2
- **Late QAT**: STE int6 fake-quantization when LR scale<0.15
- OrthoInit + muP-scaled output projections

### Quantization

- **GPTQ-lite**: Per-row optimal clip percentile search (5 candidates) for int6
- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

### Run Command

```bash
SEED=1337 bash eval/eval.sh
```

### Reproducibility

All 3 seeds produce valid artifacts under 16MB with tight variance (std=0.0005 BPB). The GPTQ-lite clip search is deterministic.
