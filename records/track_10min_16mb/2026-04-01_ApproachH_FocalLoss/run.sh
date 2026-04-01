#!/bin/bash
# Approach H: Focal Loss training run
# Base: ApproachB (Int5 GPTQ + 33.6M params) + focal loss (gamma=2.0)

set -euo pipefail

export NCCL_IB_DISABLE=1
export RUN_ID=approach_h_focal
export FOCAL_GAMMA=2.0

# All other hyperparams inherit from ApproachB defaults in Hyperparameters class
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train.log
