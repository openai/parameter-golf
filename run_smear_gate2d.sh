#!/bin/bash
# SmearGate ON + 2D attn_gate (num_heads, head_dim) with additive (1+g), init=0 (identity).
set -e

export RUN_ID="smear_gate2d_s42"
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
export SMEARGATE_ENABLED=1
export SMEARGATE_BOS_ID=1
export SMEARGATE_INIT=3.0
export ATTN_GATE_ENABLED=1
export ATTN_GATE_INIT=0.0

LOGFILE="/parameter-golf/logs/${RUN_ID}.stdout"
mkdir -p /parameter-golf/logs
cd /parameter-golf
rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_pr1493.py 2>&1 | tee "${LOGFILE}"
