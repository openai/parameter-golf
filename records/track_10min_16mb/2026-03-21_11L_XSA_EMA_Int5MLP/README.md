# 11L XSA + Continuous EMA + Int5-MLP + 8% Pruning

**val_bpb: 1.1399** (mean of 3 seeds, sliding window stride=64, post int5/int6+zstd quantization roundtrip)

## Run Command

```bash
# Setup (once)
python3 data/cached_challenge_fineweb.py --variant sp1024
pip install zstandard

# Train + evaluate
SEED=42 RUN_ID=submit_seed42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed beyond SEED and RUN_ID.

## 3-Seed Results (8xH100)

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|---------------|-------|
| 42 | 1.14005 | 15,919,150 | yes |
| 1337 | 1.13874 | 15,999,808 | yes |
| 7 | 1.14080 | 15,882,678 | yes |
| **Mean** | **1.1399** | | |
| **Std** | **0.0009** | | |

## Key Techniques

### XSA (Exclusive Self-Attention)
- Applied to last 4 of 11 layers (arXiv:2603.09078)
- Replaces standard self-attention with exclusive variant for better representation
- 11th layer funded by int5 MLP compression savings

### Continuous GPU EMA (decay=0.997)
- Exponential moving average updated every step on GPU in float32
- Replaces SWA checkpoint averaging — smoother and faster
- No CPU transfers during training (critical for step time: 102ms vs 183ms)

### Mixed Int5/Int6 Quantization + 8% Pruning
- **Int5 [-16,15]** for MLP weights (most compressible)
- **Int6 [-32,31]** for attention weights (precision-sensitive)
- **FP16** for embeddings and small tensors
- **8% magnitude pruning** post-training for compression headroom
- zstd-22 compression

### Cosine Warmdown (3000 iters)
- Time-based warmdown: cosine schedule over last ~300s of training
- Produces smoother, more compressible weight distributions

## Architecture
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(2048, dim=128)
- U-Net skip connections, tied embeddings
- Logit softcap at 30.0, RoPE base=10000

## Training Hyperparameters
- Muon optimizer: matrix_lr=0.025, scalar_lr=0.025, WD=0.04
- AdamW for embeddings: tied_embed_lr=0.035
- warmdown=3000 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, muon_momentum=0.99
- Continuous EMA: decay=0.997, every step
- Sliding window eval: stride=64

## Model Parameters
- Total: 26,829,913
- ~5,850 training steps in 600s on 8xH100

Built on techniques from PR #162 (@unnir) and PR #180 (@thwu1).
