#!/usr/bin/env bash
# FastAttn + MTP + Depth-Recurrence leaderboard run (8xH100, 10-min cap).
#
# Strategy: fork of the proven 43ms/step baseline, with three surgical upgrades:
#   (1) depth recurrence (weights shared across NUM_REPS passes)
#   (2) multi-token prediction (auxiliary loss at t+2)
#   (3) slightly bigger width (576 vs 512) since DR is param-free
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${HERE}/../../.."
TRAIN="${HERE}/train_gpt.py"

: "${DATA_PATH:=${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
: "${SEED:=42}"
: "${RUN_ID:=fastattn_mtp_dr_$(date +%s)}"

export NCCL_IB_DISABLE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
SEED="${SEED}" \
RUN_ID="${RUN_ID}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-590}" \
ITERATIONS="${ITERATIONS:-12000}" \
WARMUP_STEPS="${WARMUP_STEPS:-30}" \
WARMDOWN_ITERS="${WARMDOWN_ITERS:-1500}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
NUM_LAYERS="${NUM_LAYERS:-7}" \
MODEL_DIM="${MODEL_DIM:-576}" \
NUM_HEADS="${NUM_HEADS:-8}" \
NUM_KV_HEADS="${NUM_KV_HEADS:-2}" \
MLP_MULT="${MLP_MULT:-2}" \
NUM_REPS="${NUM_REPS:-2}" \
MTP_WEIGHT="${MTP_WEIGHT:-0.3}" \
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}" \
TIED_EMBED_LR="${TIED_EMBED_LR:-0.05}" \
MATRIX_LR="${MATRIX_LR:-0.04}" \
SCALAR_LR="${SCALAR_LR:-0.04}" \
QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}" \
torchrun --standalone --nproc_per_node=8 "${TRAIN}"
