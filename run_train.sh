#!/bin/bash
# Run SubSixteen v2 training on 8xH100
# Usage: bash run_train.sh [RUN_ID]

set -e

RUN_NAME="${1:-subsixteen_v3}"
NGPU=$(nvidia-smi -L | wc -l)

echo "=== SubSixteen v2 Training ==="
echo "GPUs detected: $NGPU"
echo "Run ID: $RUN_NAME"
echo ""

# Copy your v2 train_gpt.py from the submission records to workspace root
# (the submission folder has the exact code that scored 1.1708)
cp records/track_10min_16mb/2026-03-19_SubSixteen/train_gpt.py ./train_gpt.py

RUN_ID="$RUN_NAME" \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=4096 \
TRAIN_BATCH_TOKENS=393216 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
QUANT_BITS=6 \
USE_ZSTD=1 \
ZSTD_LEVEL=22 \
SWA_ENABLED=1 \
SWA_EVERY=200 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=$NGPU train_gpt.py 2>&1 | tee "logs/${RUN_NAME}.log"

echo ""
echo "=== Training complete ==="
echo "Log saved to: logs/${RUN_NAME}.log"
echo "Check for: final_int8_zlib_roundtrip line in the output"
