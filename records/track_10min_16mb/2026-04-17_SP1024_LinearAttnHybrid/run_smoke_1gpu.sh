#!/usr/bin/env bash
# Quick smoke test on a single GPU with a shortened training budget.
# Verifies that the forward/backward, QAT switchover, EMA/SWA merge, and
# quantise/reload round-trip all work before spending cloud GPUs on the full run.
set -euo pipefail

: "${DATA_PATH:=./data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=./data/tokenizers/fineweb_1024_bpe.model}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="${HERE}/train_gpt.py"

DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
SEED="${SEED:-0}" \
RUN_ID="${RUN_ID:-smoke}" \
MAX_WALLCLOCK_SECONDS=120 \
ITERATIONS=160 \
WARMUP_STEPS=5 \
WARMDOWN_START=120 \
WARMDOWN_END=160 \
VAL_LOSS_EVERY=80 \
TRAIN_LOG_EVERY=20 \
VAL_BATCH_SIZE=32768 \
TRAIN_BATCH_TOKENS=32768 \
TRAIN_SEQ_LEN=512 \
NUM_LAYERS=4 \
MODEL_DIM=256 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MIXER_LAYERS="1" \
SWA_START=130 \
SWA_INTERVAL=10 \
QAT_START=120 \
EVAL_STRIDE=64 \
EVAL_CTX=512 \
torchrun --standalone --nproc_per_node=1 "${TRAIN}"
