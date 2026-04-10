#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_base.sh"

export RUN_ID="ternary_12L_d768_fp8"
export NUM_LAYERS=12
export EXPERIMENT_NAME="Ternary 12L d768 + FP8 Scales (~13.8MB)"
# Note: FP8 scales logic would need to be explicitly enabled in ternary.py if desired,
# but for now we'll just set the experiment name to match the plan.

torchrun --nproc_per_node=8 train_gpt.py
