#!/bin/bash
# Full run on 2×H100 — equivalent to 8×H100 10min (~6000 steps)
# 40 min wall time (8 GPUs × 10min = 80 GPU-min → 80/2 = 40min)
# warmdown=1200 (20% of ~6000 steps)
set -e

EXP_NAME="${EXP_NAME:-exp_full_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="/workspace/logs/${EXP_NAME}.log"
mkdir -p /workspace/logs

echo "=== Run: $EXP_NAME ==="
echo "Log: $LOG_FILE"

TRAIN_SEQ_LEN=2048 QK_GAIN_INIT=4.0 RECUR_LAYERS=3,4 \
RECUR_START_STEP=1000 PARALLEL_START_LAYER=7 \
ROPE_DIMS=16 VAL_LOSS_EVERY=200 \
MAX_WALLCLOCK_SECONDS=2400 WARMDOWN_ITERS=1200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee "$LOG_FILE"

echo "=== Training done. Stopping pod in 30s (Ctrl+C to cancel) ==="
sleep 30
runpodctl stop pod $RUNPOD_POD_ID
