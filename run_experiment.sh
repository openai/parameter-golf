#!/bin/bash
# Quick experiment runner for local RTX 3060 development
# Usage: ./run_experiment.sh <experiment_name> [extra_env_vars...]
# Example: ./run_experiment.sh baseline_test NUM_LAYERS=12 MODEL_DIM=384

set -euo pipefail

EXPERIMENT_NAME="${1:?Usage: ./run_experiment.sh <experiment_name> [ENV=VAL ...]}"
shift

# Defaults for local development (RTX 3060, 12GB VRAM)
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export RUN_ID="${EXPERIMENT_NAME}"
export ITERATIONS="${ITERATIONS:-500}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-32768}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-100}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32768}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"

# Apply any extra env vars passed as arguments
for arg in "$@"; do
    export "$arg"
done

RESULTS_DIR="experiments/${EXPERIMENT_NAME}"
mkdir -p "$RESULTS_DIR"

# Log the configuration
echo "=== Experiment: ${EXPERIMENT_NAME} ===" | tee "${RESULTS_DIR}/config.txt"
echo "Date: $(date -Iseconds)" | tee -a "${RESULTS_DIR}/config.txt"
env | grep -E '^(NUM_|MODEL_|VOCAB|TRAIN_|VAL_|ITERATIONS|MLP_|TIE_|EMBED_|HEAD_|MATRIX_|SCALAR_|MUON_|WARMDOWN|WARMUP|MAX_WALL|SEED|LOGIT_|ROPE_|QK_|GRAD_|WEIGHT_DECAY|ORTHO_INIT|BIGRAM_|SWA_|QAT|SMEAR_GATE_|COMPRESSOR|QUANT_|EVAL_)' | sort | tee -a "${RESULTS_DIR}/config.txt"
echo "" | tee -a "${RESULTS_DIR}/config.txt"

# Detect the training script to use
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_gpt.py}"

# Run training (single GPU via torchrun for compatibility)
echo "Starting training..."
START_TIME=$(date +%s%N)

torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT" 2>&1 | tee "${RESULTS_DIR}/train.log"

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
echo "" | tee -a "${RESULTS_DIR}/train.log"
echo "Total wall time: ${ELAPSED_MS}ms" | tee -a "${RESULTS_DIR}/train.log"

# Extract final metrics
echo "" | tee -a "${RESULTS_DIR}/results.txt"
echo "=== Results ===" | tee "${RESULTS_DIR}/results.txt"
grep -E "(val_loss|val_bpb|final_int8|submission size|model_params)" "${RESULTS_DIR}/train.log" | tail -10 | tee -a "${RESULTS_DIR}/results.txt"

echo ""
echo "Results saved to ${RESULTS_DIR}/"
