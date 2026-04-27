#!/bin/bash
# Final 3-seed evaluation run with all best improvements combined.
# Run on 8xH100 SXM.
set -euo pipefail

NPROC="${NPROC:-8}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"

export DATA_DIR="./data/"
export VOCAB_SIZE=8192
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432

# Tuned hyperparameters (update after sweep)
export QK_GAIN_INIT=5.5
export MUON_WD=0.100
export EMA_DECAY=0.997
export MATRIX_LR=0.022
export WARMDOWN_FRAC=0.72
export WARMUP_STEPS=20

# Architecture
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35
export PARALLEL_RESIDUAL_START=7
export SKIP_GATES_ENABLED=1
export XSA_LAST_N=11
export ROPE_DIMS=16
export LN_SCALE=1

# Mixed-precision quantization
export MIXED_PRECISION_QUANT=1
export QUANT_HIGH_BITS=8
export QUANT_MID_BITS=6
export QUANT_LOW_BITS=5
export QUANT_HIGH_FRAC=0.15
export QUANT_LOW_FRAC=0.25
export QUANT_HIGH_CLIP_SIGMAS=25.0
export QUANT_MID_CLIP_SIGMAS=12.85
export QUANT_LOW_CLIP_SIGMAS=8.0

# Eval + TTT
export SLIDING_WINDOW_ENABLED=1
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export TTT_MOMENTUM=0.9
export TTT_CHUNK_TOKENS=32768

# LoRA TTT (enable after validation)
export TTT_LORA_ENABLED=1
export TTT_LORA_RANK=8
export TTT_LORA_LAYERS=4
export TTT_LORA_EPOCHS=6

# Compression
export COMPRESSOR=brotli

RESULTS_DIR="${SCRIPT_DIR}/final_results"
mkdir -p "$RESULTS_DIR"

echo "=== Running 3-seed final evaluation ==="

for seed in 42 314 999; do
    echo "--- Seed $seed ---"
    SEED=$seed RUN_ID="final_seed${seed}" \
        torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
        2>&1 | tee "${RESULTS_DIR}/train_seed${seed}.log"
done

echo "=== 3-seed evaluation complete ==="
echo "Check ${RESULTS_DIR}/ for logs"
echo "Grep for 'val_bpb' in logs to see results"
