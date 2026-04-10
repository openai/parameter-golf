#!/bin/bash
# Common architectural and training parameters for ternary experiments

export TERNARY_ENABLED=1
export QAT_BITS=0 # Ternary uses its own logic
export OPTIMIZER=muon_adam
export LR_SCHEDULE=trapezoid
export USE_FLASHATTENTION3=1

# Architecture (78.7M params, ~14.4 MB with Ternary+LZMA)
export NUM_LAYERS=12
export MODEL_DIM=768
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4

# Training
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export ITERATIONS=6500
export MAX_WALLCLOCK_SECONDS=600

# NeoMuon specific
export MUON_BACKEND_STEPS=3

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
