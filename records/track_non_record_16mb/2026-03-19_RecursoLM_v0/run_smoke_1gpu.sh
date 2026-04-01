#!/usr/bin/env bash
set -euo pipefail

# Smoke run for quick validation on a single CUDA GPU.
RUN_ID=${RUN_ID:-recurso_v0_smoke}
DATA_PATH=${DATA_PATH:-./data/datasets/fineweb10B_sp1024}
TOKENIZER_PATH=${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}

RUN_DIR="records/track_non_record_16mb/2026-03-19_RecursoLM_v0"

RUN_ID="$RUN_ID" \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=512 \
MODEL_DIM=384 \
NUM_LAYERS=2 \
RECURRENCE_STEPS=16 \
NUM_HEADS=4 \
NUM_KV_HEADS=1 \
MLP_DIM=1024 \
ITERATIONS=${ITERATIONS:-400} \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-262144} \
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600} \
torchrun --standalone --nproc_per_node=1 "$RUN_DIR/train_gpt.py"
