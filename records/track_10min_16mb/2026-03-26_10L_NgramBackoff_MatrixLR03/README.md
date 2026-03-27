# Record: 10L + Multi-Order N-gram Backoff + Matrix LR 0.03

**val_bpb = 0.9074** (3-seed mean, std 0.0002) | **15.32-15.46 MB** | 8xH100 SXM, 600s

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **N-gram BPB** | Artifact |
|------|-------|---------|---------------|----------------|----------|
| 42 | 6,693 | 89.6 | 1.1528 | **0.9076** | 15,320,749 |
| 1337 | 6,605 | 90.9 | 1.1521 | **0.9072** | 15,261,004 |
| 2024 | 6,607 | 90.8 | 1.1520 | **0.9074** | 15,457,538 |
| **Mean** | | | | **0.9074 ± 0.0002** | |

## Key Change from PR #802

Single change: **MATRIX_LR=0.03** (from 0.02). This was discovered through systematic RTX4500 screening (20 experiments) to be the single largest training hyperparameter improvement for 10L architectures (-0.064 BPB on RTX4500 screening, -0.005 BPB on 8xH100 full run).

## Architecture (same as PR #802)

- 10L, 512d, GQA 8H/4KV, MLP 3x LeakyReLU(0.5)²
- BigramHash(4096, dim=128), SmearGate, Value Residual, Gated Attention
- XSA last 4 layers, Partial RoPE 16/64, LN Scale
- U-Net skip connections, tied embeddings, logit softcap=30

## Training

- Muon optimizer: **lr=0.03** (was 0.02), momentum 0.92→0.99, WD=0.04
- EMA(0.997), warmdown=3500 steps
- Mixed int5-MLP/int6-attn quantization + zstd-22
- 3% magnitude pruning

## Eval: Multi-Order N-gram Backoff (from PR #802)

- Score-first backward-looking n-gram cache (orders 2-7)
- Highest matching order wins (backoff from 7-gram to bigram)
- Entropy-adaptive alpha: `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))`
- 4M XOR-hash buckets, min_count=2
- **Legal:** each token scored BEFORE cache is updated

## Reproduction

```bash
MATRIX_LR=0.03 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Based On

- PR #802 (@[author]): 10L + Multi-Order N-gram Backoff (0.9123 BPB)
- Our systematic hyperparameter screening (steps 10-12, 74 experiments)
