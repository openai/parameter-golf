#!/bin/bash
# ---------------------------------------------------------------------------
# Parameter Golf v2 -- 8xH100 Training Run
# Usage: bash run_h100.sh [SEED]
# Matches SOTA config: SP8192 + 3-Layer Recurrence + ParResid + QK5.25 + TTT
# ---------------------------------------------------------------------------
set -e

SEED=${1:-42}

echo "============================================"
echo " Parameter Golf v2 -- 8xH100 Run (seed=$SEED)"
echo "============================================"

SEED=$SEED \
MAX_WALLCLOCK_SECONDS=599 \
ITERATIONS=20000 \
TRAIN_BATCH_TOKENS=786432 \
VAL_BATCH_TOKENS=524288 \
VAL_LOSS_EVERY=4000 \
TRAIN_LOG_EVERY=500 \
WARMUP_STEPS=20 \
SLIDING_WINDOW_ENABLED=1 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
GPTQ_RESERVE_SECONDS=12 \
GPTQ_CALIBRATION_BATCHES=64 \
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
