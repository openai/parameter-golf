# 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04

**val_bpb: 1.14276** (mean of 3 seeds, sliding window stride=64, post int5/int6+zstd quantization roundtrip)

## Run Command

```bash
# Setup (once): downloads tokenizer + training/val shards (default 80 train shards)
bash prepare.sh

# Train + evaluate (default seed=42)
bash eval/eval.sh

# With specific seed
SEED=42 bash eval/eval.sh

# Quick smoke test (few steps, small batch; set SWA_ENABLED=0) — not a leaderboard score
bash eval/smoke.sh
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed for a full run.

For a smaller download while iterating locally, run `TRAIN_SHARDS=1 bash prepare.sh` from this directory (or pass `--train-shards 1` to `data/cached_challenge_fineweb.py` from the repo root).

## 3-Seed Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|---------------|-------|
| 42 | 1.14271 | 15,965,978 | yes |
| 1337 | 1.14298 | 15,830,186 | yes |
| 2024 | 1.14260 | ~15.8M | yes |
| **Mean** | **1.14276** | | |
| **Std** | **0.00016** | | |

## Key Techniques

### Mixed Int5/Int6 Quantization
- **Int5 [-16,15]** for MLP weights (most compressible, 1.88x zstd ratio)
- **Int6 [-32,31]** for attention weights (precision-sensitive, 1.51x zstd ratio)
- **FP16** for tied embeddings and last-layer key projections
- Int5 MLP saves ~1.86MB vs uniform int6, funding a 10th layer

### BigramHash(10240)
- Hash consecutive token pairs into 10240-bucket embedding table (dim=128)
- Projected to model_dim=512 via learned linear
- Reduces token-pair hash collisions vs 4096 buckets (+0.001 bpb)

### SWA with start_frac=0.4
- Collect checkpoints only from last 40% of warmdown (most converged)
- 24 checkpoints averaged every 50 steps
- Quality over quantity: fewer but better-converged checkpoints

## Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Training Hyperparameters
- Muon optimizer: matrix_lr=0.02, WD=0.04, momentum=0.99
- AdamW for embeddings/scalars: WD=0.04
- warmdown=3000 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- SWA: start_frac=0.4, every=50 steps
- Sliding window eval: stride=64

## Ablation Summary
| Change | val_bpb | Delta |
|--------|---------|-------|
| 9L int6 (PR162 base) | 1.1485 | baseline |
| + int5 MLP + 10th layer | 1.1453 | -0.003 |
| + WD=0.04 + warmdown=3000 | 1.1452 | -0.0001 |
| + SWA_start_frac=0.4 | 1.1446 | -0.0006 |
| + bigram=8192 | 1.1434 | -0.0012 |
| + bigram=10240 | **1.1426** | **-0.0008** |

Built on PR #162 by @unnir (SmearGate, BigramHash, OrthoInit).
