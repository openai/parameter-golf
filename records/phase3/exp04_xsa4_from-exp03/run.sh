#!/bin/bash
# ============================================================
# Exp04: XSA on last 4 layers — from Exp03
# Exclusive Self-Attention: subtract self-value from attention
# output, forcing reliance on cross-token information.
# Critical synergy with EMA (exp03).
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="exp04_xsa4_from-exp03"
cd /workspace/parameter-golf
export EVAL_STRIDE=0
export SEED="${SEED:-42}"
export ITERATIONS="${ITERATIONS:-20000}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-100}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-25}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export NUM_LAYERS=11
export UNIQUE_LAYERS=10
export MLP_ACTIVATION=leaky_relu_sq
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MOMENTUM_CYCLIC=1
export MOMENTUM_MIN=0.85
export MOMENTUM_MAX=0.95
export MOMENTUM_CYCLE_PERIOD=50
export AWQ_ENABLED=1
export AWQ_ALPHA=0.5
export ROPE_FRAC=0.25
export EMA_ENABLED=1
export EMA_DECAY=0.997
export XSA_LAST_N=4
export RUN_ID="${EXP_NAME}"
echo "=== ${EXP_NAME} ==="
python3 "${SCRIPT_DIR}/train_gpt.py" 2>&1 | tee "${SCRIPT_DIR}/logs.txt"
echo "=== ${EXP_NAME} COMPLETE ==="
