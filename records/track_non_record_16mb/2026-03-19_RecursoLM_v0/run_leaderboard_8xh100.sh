#!/usr/bin/env bash
set -euo pipefail

# Intended leaderboard-style launch on 8xH100.
RUN_ID=${RUN_ID:-recurso_v0_8xh100}
DATA_PATH=${DATA_PATH:-./data/datasets/fineweb10B_sp1024}
TOKENIZER_PATH=${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}

RUN_DIR="records/track_non_record_16mb/2026-03-19_RecursoLM_v0"

RUN_ID="$RUN_ID" \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=512 \
MODEL_DIM=${MODEL_DIM:-384} \
NUM_LAYERS=${NUM_LAYERS:-2} \
RECURRENCE_STEPS=${RECURRENCE_STEPS:-16} \
NUM_HEADS=${NUM_HEADS:-4} \
NUM_KV_HEADS=${NUM_KV_HEADS:-1} \
MLP_DIM=${MLP_DIM:-1024} \
ITERATIONS=${ITERATIONS:-20000} \
VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-1000} \
TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-524288} \
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600} \
torchrun --standalone --nproc_per_node=8 "$RUN_DIR/train_gpt.py"
