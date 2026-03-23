# PR #287 family rerun at 585s wallclock (XSA4 + EMA + BigramHash + FA3)

**Mean val_bpb: 1.1346** (3 seeds, one-sided t-test `p = 0.0032` vs merged `1.1428` SOTA)

## Summary

This is a submission-quality rerun of the PR #287 family on 8×H100 SXM with FlashAttention 3.

Recipe:
- 11 layers, 512 dim, 8 heads / 4 KV heads
- `MLP_MULT=3.0`
- `BIGRAM_VOCAB_SIZE=2048`, `BIGRAM_DIM=128`
- `XSA_LAST_N=4`
- `EMA_ENABLED=1`, `EMA_DECAY=0.997`
- `QAT_ENABLED=1`
- `EVAL_STRIDE=64`
- int6 + zstd export

The only material run-control change vs the usual 600s cap is:
- `MAX_WALLCLOCK_SECONDS=585`

That shorter wallclock keeps all three seeds under the 16,000,000-byte artifact cap while preserving merged-SOTA-beating scores.

## Exact launch command

```bash
SEED=<seed> \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=585 WARMDOWN_ITERS=3000 \
WARMUP_STEPS=20 TRAIN_LOG_EVERY=200 VAL_LOSS_EVERY=0 \
EVAL_STRIDE=64 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
NUM_LAYERS=11 MLP_MULT=3.0 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
QAT_ENABLED=1 GRAD_CLIP_NORM=0.3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact |
|---|---:|---:|---:|---:|---:|
| 1337 | 1.91585177 | 1.13467834 | 6757 | 86.58 | 15,399,663 |
| 42   | 1.91753791 | 1.13567697 | 6732 | 86.90 | 15,580,753 |
| 2025 | 1.91365474 | 1.13337713 | 6763 | 86.50 | 15,398,721 |
| **Mean** | **1.91568147** | **1.13457748** | | | |

- Max artifact size: **15,580,753 bytes**
- Code size: **67,137 bytes**
- Mean improvement vs merged `1.1428` SOTA: **-0.0082 bpb**
- One-sided t-test vs merged `1.1428`: **p = 0.0032**

## Files
- `train_gpt.py` — exact standalone script used for the runs
- `train.log` — canonical seed 1337 run
- `train_seed42.log`
- `train_seed2025.log`
- `submission.json` — metadata summary
