#!/usr/bin/env bash
set -euo pipefail

# BANDIT_WAGON single-GPU proxy winddown sweep wrapper.
# Directional signal only (not absolute parity with 8xGPU scores).
#
# Usage:
#   MODEL_PATH=/abs/path/to/final_model.pt \
#   SEEDS=444 \
#   bash experiments/Bandit_Wagon/winddown_ab_gpu1.sh
#
# Override any preset by exporting env vars before calling.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 1-GPU runtime defaults
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export SEEDS="${SEEDS:-${SEED:-1337}}"
export WINDDOWN_ITERATIONS="${WINDDOWN_ITERATIONS:-300}"
export WINDDOWN_WALLCLOCK_SECONDS="${WINDDOWN_WALLCLOCK_SECONDS:-90}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export COMPILE_ENABLED="${COMPILE_ENABLED:-0}"

# Default to a reduced arm set for faster proxy ranking.
# Set ARM_FILTER='' to run all arms.
export ARM_FILTER="${ARM_FILTER:-A_control_live|B_ema_only|D_ema_distill24|F_ema_ttt_e1_lr005}"

echo "============================================"
echo "  BANDIT_WAGON 1-GPU proxy winddown sweep"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  SEEDS=${SEEDS}"
echo "  ITERATIONS=${WINDDOWN_ITERATIONS}"
echo "  MAX_WALLCLOCK_SECONDS=${WINDDOWN_WALLCLOCK_SECONDS}"
echo "  TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS}"
echo "  VAL_BATCH_SIZE=${VAL_BATCH_SIZE}"
echo "  TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN}"
echo "  EVAL_SEQ_LEN=${EVAL_SEQ_LEN}"
echo "  ARM_FILTER=${ARM_FILTER}"
echo "============================================"

bash "${SCRIPT_DIR}/winddown_ab.sh"

