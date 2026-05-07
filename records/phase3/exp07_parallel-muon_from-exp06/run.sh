#!/bin/bash
# ============================================================
# Exp07: Parallel Muon (batched Newton-Schulz) — from Exp06
# Groups weight matrices by shape for batched NS5 orthogonalization.
# Critical for 8xH100 throughput (83ms/step target).
# On 1xGPU: validates correctness, minor speedup from batching.
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="exp07_parallel-muon_from-exp06"
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
export QAT_THRESHOLD=0.1
export USE_GPTQ=1
export GPTQ_DAMP=0.01
export GPTQ_CALIB_BATCHES=8
export RUN_ID="${EXP_NAME}"
echo "=== ${EXP_NAME} ==="
python3 "${SCRIPT_DIR}/train_gpt.py" 2>&1 | tee "${SCRIPT_DIR}/logs.txt"
echo "=== ${EXP_NAME} COMPLETE ==="
