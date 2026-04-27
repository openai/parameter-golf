# 10L MLP3x + Muon Optimizer (Single GPU Colab)

**val_bpb: 1.3365** | **1× Colab GPU** | Non-record submission

## Overview

Non-record submission exploring increased depth (10 layers vs 9 baseline) and 3× MLP expansion with Muon optimizer on a single Colab GPU.

## Results

| Config | val_loss | val_bpb | Steps | Hardware |
|--------|----------|---------|-------|----------|
| Baseline | ~2.07 | ~1.224 | 3000 | 8×H100 SXM |
| This submission | 2.2566 | 1.3365 | 3000 | 1× Colab GPU |

Higher BPB is expected — single GPU with limited compute, not 8×H100.

## Changes from Baseline

- **10 layers** (from 9)
- **3× MLP expansion** (from 4×), following top submissions
- **Muon optimizer** for matrix parameters
- **2048 sequence length**
- **20,000 max iterations**, warmup=20

## Run Command
```bash
RUN_ID=10l_mlp3x_muon \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=20000 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
TRAIN_SEQ_LEN=2048 \
WARMUP_STEPS=20 \
TRAIN_BATCH_TOKENS=393216 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Next Steps

Planning: int6 QAT, sliding window eval, EMA, LeakyReLU(0.5)², XSA

## Author

- **Name**: Durlabh Kumar Jha
- **GitHub**: durlabhkumarjha
