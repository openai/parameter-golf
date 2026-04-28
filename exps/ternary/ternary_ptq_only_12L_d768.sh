#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_base.sh"

export RUN_ID="ternary_ptq_only_12L_d768"
export NUM_LAYERS=12
export TERNARY_ENABLED=0
export PTQ_BITS=2 # Approximate ternary as 2-bit for PTQ path
export EXPERIMENT_NAME="Ternary PTQ-only 12L d768"

torchrun --nproc_per_node=8 train_gpt.py
