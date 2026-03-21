# 11L MLP3x + Int6 QAT + XSA + EMA + BigramHash + FA3

**Best val_bpb: 1.1345** (seed 2025), **mean: 1.1364** (3 seeds)

## Summary

This submission builds on the proven PR #287 recipe (11-layer, 3× MLP, int6 STE QAT, XSA, EMA, BigramHash, sliding-window eval) and runs it with FlashAttention 3 on 8×H100 SXM.

Key components:
- **11 layers**, width 512, GQA (8 heads / 4 KV heads)
- **3× MLP expansion** with SwiGLU
- **Int6 STE QAT** with zstd-22 compression
- **XSA** (cross-sequence attention) on last 4 layers
- **EMA** (decay=0.997) for weight averaging
- **BigramHash(2048)** embedding augmentation
- **Muon optimizer** (momentum=0.99, WD=0.04) + Adam for non-matrix params
- **Sliding-window eval** at stride=64
- **FlashAttention 3** (Hopper build) — required for competitive throughput

## Results

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact |
|---|---:|---:|---:|---:|---:|
| 2025 | 1.9156 | **1.1345** | 6810 | 88.10 | 15,369,315 |
| 42   | 1.9195 | 1.1368 | 6136 | 97.78 | 15,248,694 |
| 1337 | 1.9190 | 1.1379 | 6366 | 94.24 | 15,579,821 |
| **Mean** | **1.9180** | **1.1364** | | | |

All artifacts under the 16,000,000-byte cap.

## Requirements

- FlashAttention 3 (Hopper branch): `cd /tmp && git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention/hopper && pip install .`
- `pip install sentencepiece zstandard`

## Command

```bash
SEED=2025 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=3000 \
WARMUP_STEPS=20 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=0 \
EVAL_STRIDE=64 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
NUM_LAYERS=11 \
MLP_MULT=3.0 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=128 \
XSA_LAST_N=4 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
SWA_ENABLED=0 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
QAT_ENABLED=1 \
GRAD_CLIP_NORM=0.3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included files

- `train_gpt.py` — exact script used for all runs
- `train.log` — seed 2025 training log (best run)
- `submission.json` — metadata
