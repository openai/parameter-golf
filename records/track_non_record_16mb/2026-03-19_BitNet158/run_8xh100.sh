#!/bin/bash
set -e
# 12x768 BitNet b1.58 MLP3x on 8xH100 — official submission run
# Run from repo root: bash records/track_non_record_16mb/2026-03-19_BitNet158/run_8xh100.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ID=bitnet_12x768_mlp3x_8xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
ITERATIONS=20000 \
NUM_LAYERS=12 \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=6 \
MLP_MULT=3 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=2048 \
VAL_LOSS_EVERY=500 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
LR_WARMUP_STEPS=50 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
ROPE_BASE=200000 \
torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_gpt.py"
