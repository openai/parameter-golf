# QAT Int5/Int6 + Backout + U-Net Skips + BigramHash(10240) + SWA50

**val_bpb: 1.1477** (seed=42, sliding window stride=64, post int5/int6+zstd quantization roundtrip, 15.94 MB)

## Run Command

```bash
pip install zstandard  # optional but recommended for better compression
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed.
Falls back to zlib if zstandard is not installed.

## Key Techniques

### Quantization-Aware Training (QAT)
During training, large weight matrices pass through a simulated quantization bottleneck using the
straight-through estimator (STE). MLP weights see int5 noise, attention weights see int6 noise.
The model learns to be robust to quantization, reducing the post-quant penalty from ~0.016 BPB
to ~0.005 BPB — roughly 0.01 BPB free compared to post-training quantization alone.

### Backout (Learned Residual Subtraction)
A learned scalar λ (init=0.2) subtracts the midpoint layer's activation from the final output:
`x = x - λ * x_mid`. Prevents over-reliance on early representations.

### U-Net Skip Connections
Encoder-decoder structure (5+5 layers) with learned per-dimension skip weights connecting
encoder to decoder layers.

### SVD Embedding Initialization
Tied embeddings initialized via SVD spectral decay: singular values reshaped to follow a
1/√k profile for better initial token representations.

### Mixed Int5/Int6 Quantization + zstd-22
- Int5 [-16,15] for MLP weights (most compressible)
- Int6 [-32,31] for attention weights (precision-sensitive)
- FP16 for tied embeddings and last-2-layer key projections (Late-K)

### BigramHash(10240) + SmearGate
Hash consecutive token pairs into 10240-bucket embedding table (dim=128, projected to 512).
SmearGate blends each token with the previous token's embedding.

### Stochastic Weight Averaging
SWA every 50 steps during warmdown (start_frac=0.4). Smoother weight distributions quantize better.

## Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu² activation
- Tied embeddings, logit softcap=30
- Orthogonal init with muP-scaled output projections

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| matrix_lr (Muon) | 0.02 |
| scalar_lr (AdamW) | 0.02 |
| tied_embed_lr | 0.03 |
| muon_weight_decay | 0.04 |
| adamw_weight_decay | 0.01 |
| warmdown_iters | 3000 |
| swa_every / start_frac | 50 / 0.4 |
| prune_frac | 0.08 |
| eval_stride | 64 |
| compressor | zstd-22 |

Built on PR #162 by @unnir (SmearGate, BigramHash, OrthoInit) and techniques from @thwu1 and @raahilshah.
