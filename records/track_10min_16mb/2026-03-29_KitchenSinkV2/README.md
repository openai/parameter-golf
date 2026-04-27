# Kitchen Sink V2 — Improved

Built on PR #549 / KitchenSinkV2 with the following additions:

1. **Split early/late LR banks** — separate Muon and Adam optimizers for the first and second half of layers
2. **MiLe margin loss** — triangle-scheduled margin loss with gamma=0.75, clamp_min=0.2
3. **Cache + backout residual** — layer 7 output cached and mixed back via learnable gate
4. **LeakyReLU(0.5)²** activation in MLP
5. **XSA on last 7 layers** (up from default 4)
6. **Coprime-stride multi-shard data loader** (PR #726 / #1060 style)
7. **Train-data GPTQ int6 calibration** (PR #1060) — calibration uses training data within the training budget (14s reserved from 600s)
8. **Residual lambdas** — learnable per-sublayer residual scaling (init sqrt(1.1), 5x scalar LR, no weight decay)
9. **Bigger bigram hash** — 6144 buckets (up from 2048), reducing collision ratio
10. **Bigger value embeddings** — dim=196 on layers 5,9,10 (up from dim=128 on layers 9,10)
11. **Flash Attention 3** via flash_attn_interface
12. **Sliding window eval** with stride=64

## Results (12 seeds)

| Seed | val_loss (nats) | val_bpb |
|------|----------------|---------|
| 2 | 1.8793 | 1.1130 |
| 9999 | 1.8800 | 1.1134 |
| 22 | 1.8801 | 1.1135 |
| 7 | 1.8807 | 1.1139 |
| 1337 | 1.8808 | 1.1139 |
| 2222 | 1.8807 | 1.1139 |
| 99 | 1.8808 | 1.1139 |
| 77 | 1.8815 | 1.1143 |
| 2026 | 1.8814 | 1.1143 |
| 42 | 1.8817 | 1.1145 |
| 777 | 1.8818 | 1.1145 |
| 222 | 1.8820 | 1.1147 |

| Metric | val_loss (nats) | val_bpb |
|--------|----------------|---------|
| Mean | 1.8809 | 1.1140 |
| Std | 0.0008 | 0.0005 |

### Statistical significance

Current leader: 1.1194 bpb (~1.8901 nats).

- **Improvement: 0.0091 nats / 0.0054 bpb**
- One-sample t-test vs 0.005 nats improvement: t = -17.26, df = 11, **p < 0.0001**
- One-sample t-test vs 0.005 bpb improvement: t = -2.93, df = 11, **p = 0.007**

## Artifact size (worst-case, seed 777)

| Component | Bytes |
|-----------|-------|
| Model (int6+lzma) | 15,758,116 |
| Code | 126,292 |
| **Total** | **15,884,408** |

Under the 16,000,000 byte limit.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| MATRIX_LR (early) | 0.036 |
| MATRIX_LR (late) | 0.044 |
| SCALAR_LR (early) | 0.028 |
| SCALAR_LR (late) | 0.018 |
| TIED_EMBED_LR | 0.022 |
| TRAIN_BATCH_TOKENS | 548,864 |
| BIGRAM_VOCAB_SIZE | 6,144 |
| VE_DIM | 196 |
| VE_LAYERS | 5,9,10 |
| RESID_LAMBDA_INIT | sqrt(1.1) |
| RESID_LAMBDA_LR | 5x scalar_lr |

## Command

```bash
SEED=2 MATRIX_LR=0.036 MATRIX_LR_LATE=0.044 \
SCALAR_LR=0.028 SCALAR_LR_LATE=0.018 \
TIED_EMBED_LR=0.022 TRAIN_BATCH_TOKENS=548864 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
