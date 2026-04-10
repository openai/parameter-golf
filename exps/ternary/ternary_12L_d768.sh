#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_base.sh"

export RUN_ID="ternary_12L_d768"
export NUM_LAYERS=12
export EXPERIMENT_NAME="Ternary 12L d768 (78.7M params, ~14.4MB)"

torchrun --nproc_per_node=8 train_gpt.py
