#!/bin/zsh
set -euo pipefail

: "${RUN_ID:=mlx_mtp_sweep}"
: "${ITERATIONS:=200}"
: "${TRAIN_BATCH_TOKENS:=8192}"
: "${VAL_LOSS_EVERY:=0}"
: "${VAL_BATCH_SIZE:=8192}"
: "${MTP_NUM_HEADS:=1}"
: "${SWEEP_MTP_LOSS_WEIGHTS:=0.03,0.07,0.10,0.15}"
: "${SWEEP_MTP_ACTIVE_UNTIL_FRACTIONS:=0.6,0.8,1.0}"

export RUN_ID
export ITERATIONS
export TRAIN_BATCH_TOKENS
export VAL_LOSS_EVERY
export VAL_BATCH_SIZE
export MTP_NUM_HEADS
export SWEEP_MTP_LOSS_WEIGHTS
export SWEEP_MTP_ACTIVE_UNTIL_FRACTIONS

python3 records/track_10min_16mb/2026-04-04_train_only_MTP_experiment/train_gpt_mlx_mtp_sweep.py
