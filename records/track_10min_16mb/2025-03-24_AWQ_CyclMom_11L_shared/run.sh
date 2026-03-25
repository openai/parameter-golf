#!/bin/bash
# ============================================================
# SUBMISSION: exp18 AWQ + cyclic momentum + relu_sq + 11L shared
# 8×H100 SXM, 3 seeds, full sliding window eval
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="submission_exp18_awq-cyclic-relusq-11Lshared_8xH100"
LOG_DIR="records/h100_experiments/${EXP_NAME}/logs"

cd /workspace/parameter-golf
mkdir -p "${LOG_DIR}"

# --- Architecture ---
export NUM_LAYERS=11
export UNIQUE_LAYERS=10
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3.0
export MLP_ACTIVATION=relu_sq
export VOCAB_SIZE=1024
export TIE_EMBEDDINGS=1
export LOGIT_SOFTCAP=30.0
export BIGRAM_VOCAB_SIZE=10240
export BIGRAM_DIM=128

# --- Training (8×H100 scale) ---
export ITERATIONS=20000
export WARMUP_STEPS=20
export WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export MAX_WALLCLOCK_SECONDS=600

# --- Optimizer ---
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MOMENTUM_CYCLIC=1
export MOMENTUM_MIN=0.85
export MOMENTUM_MAX=0.95
export MOMENTUM_CYCLE_PERIOD=50
export GRAD_CLIP_NORM=0.3
export WEIGHT_DECAY=0.04

# --- SWA ---
export SWA_ENABLED=1
export SWA_START_FRAC=0.2
export SWA_EVERY=50

# --- Validation & Eval ---
export VAL_LOSS_EVERY=500
export VAL_BATCH_SIZE=524288
export TRAIN_LOG_EVERY=100
export EVAL_STRIDE=64
export EVAL_BATCH_SEQS=64

# --- AWQ ---
export AWQ_ENABLED=1
export AWQ_ALPHA=0.5

# --- Run 3 seeds ---
for SEED in 42 43 44; do
    export SEED
    export RUN_ID="${EXP_NAME}_seed${SEED}"
    echo "============================================"
    echo "=== SEED ${SEED} ==="
    echo "============================================"
    torchrun --nproc_per_node=8 "${SCRIPT_DIR}/train_gpt.py" 2>&1 | tee "${LOG_DIR}/${EXP_NAME}_seed${SEED}.log"
    echo "=== SEED ${SEED} COMPLETE ==="
    echo ""
done

echo "=== ALL 3 SEEDS COMPLETE ==="
