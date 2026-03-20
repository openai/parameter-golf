# Wavelet-Weighted Widenet (WWW)

**Track:** 10min / 16MB
**val_bpb:** 1.1719 (sliding window) / 1.1754 (TTT LoRA)
**Submission size:** 15,367,830 bytes (~14.7MB)
**Author:** dubthecat

## Core Idea

Trade **bits per parameter** for **more parameters**. Standard int8 quantization gives 8 bits/param. By using ternary weights `{-1, 0, +1}` with vector quantization, we compress MLP layers to ~1 bit/param — an 8x reduction. This budget savings lets us run a wider and deeper model (35.2M params vs 18.9M baseline) while still fitting in 16MB.

## Architecture

| Component | Baseline | WWW |
|-----------|----------|-----|
| Layers | 10 | **12** |
| MLP width ratio | 2:1 (1024) | **4:1 (2048)** |
| MLP weights | float32 → int8 | **ternary → VQ** |
| Total params | 18.9M | **35.2M (1.86x)** |
| Compressed size | ~15.5MB | **~13.8MB** |

### Model: WWWGPT

- **12 transformer layers**, 512 dim, 8 heads, 4 KV heads (GQA)
- **Encoder-decoder skip connections** (U-Net style): first 6 layers are encoder, last 6 are decoder with additive skip connections
- **Tied FP16 embeddings** with overtone spectral initialization
- **Logit softcap** at 30.0
- **Phase-transition residual mixing** (sigmoid-scheduled per layer)

### TernaryLinear MLP

Each MLP layer uses `TernaryLinear` instead of standard linear:

1. Maintain latent FP32 weights during training (for optimizer quality)
2. Forward pass: quantize to `{-1, 0, +1}` with per-row scaling
   - `alpha = mean(|W|)` per row
   - `W_ternary = clamp(round(W / alpha), -1, 1)`
   - STE (Straight-Through Estimator): gradients flow to latent weights
3. **relu² activation** (squared ReLU) between the two ternary layers
4. At export: extract raw ternary weights + scales

Based on BitNet b1.58 (Ma et al., 2024).

### Compression Pipeline

Three-tier hybrid compression:

1. **VQ for ternary MLP layers** (~1 bit/param):
   - Split weight rows into sub-vectors of length K=8
   - Build codebook of 256 most frequent patterns (uint8 indices)
   - Storage: codebook (2KB) + indices + per-row fp16 scales
   - Per-layer: ~134KB × 2 matrices = ~268KB (vs ~2MB at int8)

2. **int8 for attention layers** (~8 bits/param):
   - Standard per-row quantization with fp16 scales
   - 12 layers × ~789KB = ~9.5MB

3. **zlib** final pass on the full artifact

**Size breakdown:**
| Component | Size |
|-----------|------|
| Attention (int8) | ~9.5MB |
| MLP (ternary VQ) | ~3.2MB |
| Embeddings (FP16) | ~1.0MB |
| Overhead + zlib | remainder |
| **Total** | **~14.7MB** |

## Training

### 4-Way Optimizer Split

| Parameter group | Optimizer | LR | Notes |
|----------------|-----------|-----|-------|
| Token embedding | Adam | 0.6 | Standard embedding training |
| Attention matrices | **Muon** | 0.04 | Newton-Schulz orthogonalization |
| Ternary MLP weights | **Adam** | 0.02 | Better for STE gradient flow |
| Scalar/control params | Adam | 0.04 | RMSNorm, skip weights, etc. |

- Ternary weight decay: 0.01 (encourages zeros → better compression)
- Gradient clipping: norm=1.0 (stabilizes STE training)
- Muon momentum: 0.95 (warmup from 0.85 over 500 steps)

### Training Schedule

- Batch: 524,288 tokens/step, seq_len=1024
- Warmup: 20 steps (compilation priming)
- Warmdown: 2,500 iterations
- Wallclock cap: 600 seconds
- Reached step 12,113 before wallclock cap

## Evaluation

1. **Sliding window** (stride=64): val_bpb = **1.1719**
2. **TTT LoRA** (per-document adaptation): val_bpb = **1.1754**
   - Rank-8 LoRA on Q/V projections + LM head
   - Per-document reset, chunk_size=256, batch_size=64


