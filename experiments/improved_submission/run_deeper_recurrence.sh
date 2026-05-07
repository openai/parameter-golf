#!/bin/bash
# Deeper recurrence experiments.
# Tests 4-layer recurrence (L2-5) vs 3-layer (L3-5) baseline.
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

RESULTS_DIR="${SCRIPT_DIR}/recurrence_results"
mkdir -p "$RESULTS_DIR"

run_experiment() {
    local name="$1"
    shift
    echo "=== Running: $name ==="
    env "$@" RUN_ID="$name" SEED=42 \
        torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
        2>&1 | tee "${RESULTS_DIR}/${name}.log"
}

# Baseline: 3-layer recurrence L3-5 (current SOTA config)
run_experiment "recur_3layer_L3_5" \
    NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35

# 4-layer recurrence L2-5
run_experiment "recur_4layer_L2_5" \
    NUM_LOOPS=2 LOOP_START=2 LOOP_END=5 ENABLE_LOOPING_AT=0.30

# 4-layer recurrence L2-5, earlier activation
run_experiment "recur_4layer_L2_5_early" \
    NUM_LOOPS=2 LOOP_START=2 LOOP_END=5 ENABLE_LOOPING_AT=0.25

# 3-layer recurrence with 3 loops (more repetitions)
run_experiment "recur_3layer_3loops" \
    NUM_LOOPS=3 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35

# Two separate recurrence groups (L2-3 and L4-5)
run_experiment "recur_2layer_L4_5" \
    NUM_LOOPS=2 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.35

echo "=== Recurrence experiments complete ==="
