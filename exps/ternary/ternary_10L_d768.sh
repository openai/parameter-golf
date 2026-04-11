#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_base.sh"

export RUN_ID="ternary_10L_d768"
export NUM_LAYERS=10
export EXPERIMENT_NAME="Ternary 10L d768 (65.7M params, ~12MB)"

torchrun --nproc_per_node=8 train_gpt.py
