#!/usr/bin/env bash
# Single-GPU smoke test to verify the new arch compiles and trains a few steps.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${HERE}/../../.."
TRAIN="${HERE}/train_gpt.py"

: "${DATA_PATH:=${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

export NCCL_IB_DISABLE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
RUN_ID="smoke_$(date +%s)" \
MAX_WALLCLOCK_SECONDS=120 \
ITERATIONS=200 \
WARMUP_STEPS=10 \
WARMDOWN_ITERS=50 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=20 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=65536 \
VOCAB_SIZE=1024 \
NUM_LAYERS=4 \
MODEL_DIM=256 \
NUM_HEADS=4 \
NUM_KV_HEADS=1 \
MLP_MULT=2 \
NUM_REPS=2 \
MTP_WEIGHT=0.3 \
TIE_EMBEDDINGS=1 \
torchrun --standalone --nproc_per_node=1 "${TRAIN}"
