#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_base.sh"

export RUN_ID="ternary_12L_d768_muon5"
export NUM_LAYERS=12
export MUON_BACKEND_STEPS=5
export EXPERIMENT_NAME="Ternary 12L d768 Muon steps=5"

torchrun --nproc_per_node=8 train_gpt.py
