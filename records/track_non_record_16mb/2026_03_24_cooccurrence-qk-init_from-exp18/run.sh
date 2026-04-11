#!/bin/bash
# ============================================================
# Exp29: Co-occurrence QK Init — from Exp18
# Builds bigram co-occurrence matrix from first 2M training tokens (~1s),
# SVDs it, initializes W_Q and W_K in layer 0 so initial attention
# patterns reflect token co-occurrence structure. Zero extra params.
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="exp29_cooccurrence-qk-init_from-exp18"
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
export MLP_ACTIVATION=relu_sq
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export SWA_START_FRAC=0.2
export SWA_EVERY=50
export MOMENTUM_CYCLIC=1
export MOMENTUM_MIN=0.85
export MOMENTUM_MAX=0.95
export MOMENTUM_CYCLE_PERIOD=50
export AWQ_ENABLED=1
export AWQ_ALPHA=0.5
export COOC_QK_INIT=1
export RUN_ID="${EXP_NAME}"
echo "=== ${EXP_NAME} ==="
echo "Co-occurrence QK init: layer 0, 2M tokens"
python3 "${SCRIPT_DIR}/train_gpt.py" 2>&1 | tee "${SCRIPT_DIR}/logs.txt"
echo "=== ${EXP_NAME} COMPLETE ==="
