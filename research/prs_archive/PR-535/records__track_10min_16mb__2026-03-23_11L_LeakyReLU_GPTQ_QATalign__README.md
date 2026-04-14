# Record: 11L LeakyReLU² + Full GPTQ + QAT Alignment

**val_bpb: 1.1204** (3-seed mean, std 0.0001) | **15.85 MB** max artifact | 8xH100 SXM, 600s

## Results (3 seeds, 8xH100 SXM)

| Seed | Steps | val_loss | Sliding BPB (s64) | Artifact |
|------|-------|----------|-------------------|----------|
| 7    | ~6820 | 1.8915 | **1.1203** | 15,762,694 bytes |
| 314  | ~6820 | 1.8919 | 1.1205 | 15,732,473 bytes |
| 2024 | ~6820 | 1.8917 | 1.1204 | 15,851,228 bytes |

**Mean: 1.1204 | Std: 0.0001**

## Key Innovations

### 1. LeakyReLU(0.5)² Activation

Replace `relu(x)²` with `leaky_relu(x, 0.5)²` in the MLP. Standard relu² zeroes out negative activations, creating dead neurons whose down-projection weights are wasted capacity. LeakyReLU(0.5)² maps negatives to `(0.5x)² = 0.25x²`, allowing the down projection to see non-zero gradients from all neurons. This effectively doubles MLP capacity without adding parameters. Observable as dramatically faster early training convergence.

### 2. Full GPTQ Quantization

Hessian-aware GPTQ replaces percentile-search quantization. For each weight matrix:
- 256-sample calibration set from training data
- Per-layer Hessian approximation (H = X^T X)
- Column-wise int6 quantization with Cholesky error compensation
- Block size 128, column reordering by ascending Hessian diagonal

Reduces int6 quantization gap from 0.0085 to 0.0059 BPB (31% reduction).

### 3. QAT-Export Alignment

The STE fake-quantizer during training must match the export quantizer. We use `quantile(0.9995)` for per-row clipping in both the STE and the final export path, ensuring training optimizes against the actual quantization that will be applied.

## Architecture

- 11 transformer layers, dim=512, 8 heads, 4 KV heads (GQA)
- 3x MLP expansion (hidden=1536) with **LeakyReLU(0.5)²** activation
- XSA on last 4 layers (Exclusive Self-Attention)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- U-Net skip connections (5 encoder, 6 decoder)
- SmearGate temporal gating
- BigramHash (2048 buckets, 128-dim)
- Shared Value Embedding (dim=128, layers 9-10)
- FlashAttention 3 (Hopper native kernels)
- Orthogonal init, logit softcap 30, tied embeddings

## Training

- Muon optimizer (matrices): lr=0.025, momentum=0.99, WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3,500 iters (wallclock-based)
- EMA (decay=0.997) + Tight SWA (every 50 steps, scale<0.2)
- Late QAT: STE int6 fake-quantization when LR scale<0.15

## Quantization

- Full GPTQ with 256-sample Hessian calibration
- Int6 per-row with quantile(0.9995) clipping
- Small tensors + tok_emb.weight in fp16
- zstd level 22 compression

## Run Command

```bash
SEED=7 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Test Plan

- [x] 3 seeds run on 8xH100 SXM
- [x] All 3 seeds train in ≤600s
- [x] All 3 seeds artifact ≤16,000,000 bytes (max: 15,851,228)
- [x] Sliding window eval (stride=64) consistent (std=0.0001)
- [x] No test-time training on validation data
- [x] No network calls during evaluation
