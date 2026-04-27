# 12L + Catalytic Residuals + BigramHash(10240) + SWA + Late QAT

**val_bpb: 1.14662** (mean of 3 seeds, sliding window stride=64, post int6+zstd quantization roundtrip)

## Run Command

```bash
# Setup (once)
pip install sentencepiece zstandard
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# Train + evaluate (default seed=1337)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# With specific seed
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed.

## 3-Seed Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|---------------|-------|
| 1337 | 1.14749 | 14,014,540 | yes |
| 42 | 1.14575 | 14,104,510 | yes |
| 7 | 1.14662 | 14,385,363 | yes |
| **Mean** | **1.14662** | | |
| **Std** | **0.00071** | | |

## Key Techniques

### Catalytic Residual Connections (Novel)
- Replace `x + f(x)` with `x + c * f(x)`, where `c` is a learned per-dimension vector
- Initialized to ones (starts as standard residual)
- Provides dimension-wise gain control on residual connections
- Consistent -0.024 bpb improvement at zero computational overhead (~11K extra params)

### 12 Layers (Depth Scaling)
- Standard stack uses 10-11 layers, leaving significant budget headroom
- 12 layers validated as the depth sweet spot (-0.023 bpb vs 11L)
- 13L+ shows diminishing returns due to throughput cost

### BigramHash(10240)
- Hash consecutive token pairs into 10240-bucket embedding table (dim=128)
- Projected to model_dim=512 via learned linear
- -0.070 bpb improvement over BigramHash(2048)

### Late QAT (Quantization-Aware Training)
- STE (straight-through estimator) int6 quantization in the final 4% of training
- Forward uses quantized weights, backward gets full-precision gradients
- Closes quantization gap to ~0.015 bpb

### SWA (Stochastic Weight Averaging)
- Collect checkpoints from last 20% of warmdown
- Average ~16 checkpoints for smoother weight landscape

## Architecture
- 12 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings (FP16)
- XSA (cross-sequence attention) on last 4 layers
- Value Embeddings (dim=128) on layers 10, 11
- Partial RoPE (16/64 dims)
- LN Scale (1/sqrt(layer_idx+1))

## Training Hyperparameters
- Muon optimizer: matrix_lr=0.04, WD=0.042, momentum=0.95
- AdamW for embeddings/scalars: WD=0.042
- warmdown=4000 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- SWA: start_scale=0.2, every 50 steps
- Late QAT: threshold=0.25

## Evaluation
- Int6+zstd quantization roundtrip
- Sliding window eval: stride=64
- No TTT (test-time training)

## Training Metrics
- ~5,370 steps in 600s (~112 ms/step) on 8xH100 SXM
- Peak memory: ~25 GB per GPU

Built on the standard stack from PR #180 by @thwu1.
