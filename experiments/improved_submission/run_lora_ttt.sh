#!/bin/bash
# LoRA TTT experiments.
# Compare standard full-weight TTT vs LoRA-based TTT with different configs.
set -euo pipefail

NPROC="${NPROC:-8}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"

export DATA_DIR="./data/"
export VOCAB_SIZE=8192
export SLIDING_WINDOW_ENABLED=1
export MIXED_PRECISION_QUANT=1
export COMPRESSOR=brotli

RESULTS_DIR="${SCRIPT_DIR}/lora_ttt_results"
mkdir -p "$RESULTS_DIR"

run_experiment() {
    local name="$1"
    shift
    echo "=== Running: $name ==="
    env "$@" RUN_ID="$name" SEED=42 \
        torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
        2>&1 | tee "${RESULTS_DIR}/${name}.log"
}

# Baseline: standard TTT with SGD, 3 epochs
run_experiment "ttt_sgd_3ep" \
    TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 \
    TTT_LORA_ENABLED=0

# LoRA TTT: rank 8, last 4 layers, 6 epochs
run_experiment "lora_r8_l4_6ep" \
    TTT_ENABLED=0 TTT_LORA_ENABLED=1 TTT_LORA_RANK=8 TTT_LORA_LAYERS=4 \
    TTT_LORA_EPOCHS=6 TTT_LR=0.005

# LoRA TTT: rank 4, last 4 layers, 8 epochs (even fewer params)
run_experiment "lora_r4_l4_8ep" \
    TTT_ENABLED=0 TTT_LORA_ENABLED=1 TTT_LORA_RANK=4 TTT_LORA_LAYERS=4 \
    TTT_LORA_EPOCHS=8 TTT_LR=0.005

# LoRA TTT: rank 16, last 6 layers, 4 epochs (more capacity)
run_experiment "lora_r16_l6_4ep" \
    TTT_ENABLED=0 TTT_LORA_ENABLED=1 TTT_LORA_RANK=16 TTT_LORA_LAYERS=6 \
    TTT_LORA_EPOCHS=4 TTT_LR=0.003

# Combined: standard TTT + LoRA TTT (both enabled)
run_experiment "ttt_sgd_then_lora" \
    TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=2 TTT_MOMENTUM=0.9 \
    TTT_LORA_ENABLED=1 TTT_LORA_RANK=8 TTT_LORA_LAYERS=4 \
    TTT_LORA_EPOCHS=4 TTT_LR=0.003

echo "=== LoRA TTT experiments complete ==="
