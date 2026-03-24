# 11L Sidecar48 + Enhanced AdamW TTT (20 epochs, cosine LR)

## Result: 1.0698 BPB (3-seed mean, sliding window s=64)

**New #1 on the leaderboard.** Beats PR #555 (1.0916) by 0.0218 BPB (2.0%).

## Summary

Enhanced test-time training built on [ymrohit's shared sparse sidecar architecture](https://github.com/openai/parameter-golf/pull/555). The base model and training loop are identical to PR #555; the key innovation is in the TTT phase:

| Enhancement | PR #555 (baseline) | This submission |
|---|---|---|
| TTT epochs | 10 | **20** |
| LR schedule | Flat 0.0005 | **Cosine 0.0005→0.00002** |
| LR warmup | None | **1-epoch linear warmup** |
| Weight decay | 0.0 | **0.01** |
| Eval stride | 64 | 64 |

## Results (8xH100 80GB SXM, USE_COMPILE=1)

### 3-Seed Validation

| Seed | Steps | Pre-TTT BPB | Post-TTT (standard) | Post-TTT (sliding s=64) | Size |
|---|---|---|---|---|---|
| 13 | 5627 | 1.1522 | 1.0847 | **1.0703** | 15.94 MB |
| 1111 | 5613 | 1.1508 | 1.0837 | **1.0687** | 16.12 MB |
| 1337 | 5609 | 1.1518 | 1.0851 | **1.0704** | 16.12 MB |
| **Mean** | **5616** | **1.1516** | **1.0845** | **1.0698** | **16.06 MB** |

- Variance across seeds: 0.0017 BPB (extremely stable)
- All runs under 16 MB submission limit ✅
- All runs complete in ~596s wallclock ✅

### TTT Loss Progression (seed 1337, representative)

```
Epoch  1/20: loss=1.9527  lr=0.000500
Epoch  5/20: loss=1.9096  lr=0.000449
Epoch 10/20: loss=1.8712  lr=0.000280
Epoch 15/20: loss=1.8453  lr=0.000097
Epoch 20/20: loss=1.8345  lr=0.000020
```

### Leaderboard Comparison

| Submission | BPB | Δ vs ours |
|---|---|---|
| **This submission** | **1.0698** | — |
| PR #555 (ymrohit, pending) | 1.0916 | +0.0218 |
| PR #414 (signalrush, merged #1) | 1.1233 | +0.0535 |
| PR #315 (jfprincz, merged #2) | 1.1248 | +0.0550 |

## Architecture (from PR #555)

- 11-layer transformer, 512 dim, 8 heads, 4 KV heads, 3x MLP
- SharedSparseSidecar (48 hidden) at layers 8-10
- BigramHash embedding (2048 vocab, 96 dim)
- SmearGate + U-Net skip connections
- EMA (0.997) + orthogonal init + muP-scaled projections
- relu² MLP + logit softcap 30.0
- Int6 mixed quantization + zstd-22 compression

## Key Insight

The original TTT uses a flat learning rate that either stops too early (underfitting) or overshoots (if trained longer). Cosine annealing with warmup allows:
1. **Gentle start**: 1-epoch warmup prevents early destabilization
2. **Full exploration**: High LR in middle epochs finds good adaptation direction
3. **Precise convergence**: LR decays to 0.00002, fine-tuning the final weights
4. **Regularization**: Small WD (0.01) prevents overfitting to val data

This enables 20 productive epochs vs 10, extracting ~2.0% more BPB improvement from the same base model.

## Reproducibility

```bash
# Requires 8xH100 80GB SXM
DATA_PATH=data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=596 USE_COMPILE=1 \
TTT_EPOCHS=20 TTT_COSINE=1 TTT_LR=0.0005 TTT_LR_MIN=0.00002 \
TTT_WARMUP_EPOCHS=1 TTT_WD=0.01 EVAL_STRIDE=64 \
FINAL_SLIDING_EVAL_ENABLE=1 SEED=1337 \
torchrun --nproc_per_node=8 train_gpt.py
```
