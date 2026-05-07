#!/bin/bash
# SwiGLU experiment: compare SwiGLU MLP vs LeakyReLU^2.
set -euo pipefail

NPROC="${NPROC:-8}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"

export DATA_DIR="./data/"
export VOCAB_SIZE=8192
export SLIDING_WINDOW_ENABLED=1
export MIXED_PRECISION_QUANT=1
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export COMPRESSOR=brotli

RESULTS_DIR="${SCRIPT_DIR}/swiglu_results"
mkdir -p "$RESULTS_DIR"

run_experiment() {
    local name="$1"
    shift
    echo "=== Running: $name ==="
    env "$@" RUN_ID="$name" SEED=42 \
        torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
        2>&1 | tee "${RESULTS_DIR}/${name}.log"
}

# Baseline: LeakyReLU^2 MLP 4x (current SOTA)
run_experiment "leakyrelu2_4x" USE_SWIGLU=0 MLP_MULT=4

# SwiGLU MLP 2.67x (matched param count to LeakyReLU^2 4x)
run_experiment "swiglu_2.67x" USE_SWIGLU=1 SWIGLU_MLP_MULT=2.67

# SwiGLU MLP 3x (slightly more params)
run_experiment "swiglu_3x" USE_SWIGLU=1 SWIGLU_MLP_MULT=3.0

# SwiGLU MLP 2.5x (slightly fewer params, more room for quant)
run_experiment "swiglu_2.5x" USE_SWIGLU=1 SWIGLU_MLP_MULT=2.5

echo "=== SwiGLU experiments complete ==="
