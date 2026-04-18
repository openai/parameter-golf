#!/bin/bash
# Quick 10-min test on 2×H100
# ~1500 steps (8×H100 gets ~6000 in 10min, 2× gets 1/4 of that)
# warmdown=300 (20% of ~1500)
set -e

EXP_NAME="${EXP_NAME:-exp_10m_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="/workspace/logs/${EXP_NAME}.log"
mkdir -p /workspace/logs

echo "=== Run: $EXP_NAME ==="
echo "Log: $LOG_FILE"

TRAIN_SEQ_LEN=2048 QK_GAIN_INIT=4.0 RECUR_LAYERS=3,4 \
RECUR_START_STEP=1000 PARALLEL_START_LAYER=7 \
ROPE_DIMS=16 VAL_LOSS_EVERY=200 \
MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=300 \
torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee "$LOG_FILE"

echo "=== Training done. Stopping pod in 30s (Ctrl+C to cancel) ==="
sleep 30
runpodctl stop pod $RUNPOD_POD_ID
