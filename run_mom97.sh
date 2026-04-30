#!/bin/bash
# Stack MUON_MOMENTUM=0.97 (PR #1514) on top of wd_strong_paired baseline.
# Baseline (wd_strong_paired single seed): q_ttt = 1.07971
# Target: any improvement over 1.07971 from changing 0.99 -> 0.97 final momentum.
set -e

export RUN_ID="mom97_s42"
export SEED=42
export DATA_DIR=/workspace/data/
export QK_GAIN_INIT=5.25
export TTT_ENABLED=1
export TTT_LR=0.007
export TTT_EPOCHS=5
export WD_SCHEDULE_ENABLED=1
export PAIRED_HEAD_MUON_ENABLED=1
export WD_SCHED_LOW_FACTOR=0.50
export WD_SCHED_HIGH_FACTOR=1.75
export MUON_MOMENTUM=0.97

LOGFILE="/parameter-golf/logs/${RUN_ID}.stdout"
mkdir -p /parameter-golf/logs

echo "========================================"
echo "  ${RUN_ID}: MUON_MOMENTUM=0.97 on wd_strong_paired"
echo "========================================"

cd /parameter-golf
rm -rf ~/.cache/torch_extensions 2>/dev/null || true
./safe_launch.sh torchrun --standalone --nproc_per_node=8 train_pr1493.py 2>&1 | tee "${LOGFILE}"

echo "========================================"
echo "  ${RUN_ID} DONE  -- check logs/${RUN_ID}.txt for q_ttt"
echo "========================================"
grep -E "quantized_ttt|quantized val_bpb|Total submission" "/parameter-golf/logs/${RUN_ID}.txt" || true
