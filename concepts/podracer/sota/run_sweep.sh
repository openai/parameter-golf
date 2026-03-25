#!/bin/bash
set -euo pipefail
# N-gram parameter sweep — eval only, no training
# Loads existing quantized model and tests ~25 param combos
#
# REQUIRES: a podracer model trained first. Either:
#   1. Run run.sh first to train + quantize, OR
#   2. Point MODEL_PATH to a saved .int6.ptz file
#
# Each combo takes ~2-3 min on 8xH100. Total sweep: ~60-80 min.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-final_model.int6.ptz}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SWEEP_MAX_SECONDS="${SWEEP_MAX_SECONDS:-180}"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: Model file not found: ${MODEL_PATH}"
    echo "Train a podracer first (run.sh) or set MODEL_PATH"
    exit 1
fi

echo "============================================"
echo "  N-GRAM PARAMETER SWEEP"
echo "  Model: ${MODEL_PATH}"
echo "  Per-combo budget: ${SWEEP_MAX_SECONDS}s"
echo "  GPUs: ${NPROC_PER_NODE}"
echo "============================================"

# Architecture params must match the model that was trained
SEED="${SEED:-1337}" \
MLP_ACT=leaky_relu_sq \
MLP_LEAKY_SLOPE=0.5 \
XSA_LAST_N=4 \
BIGRAM_VOCAB_SIZE=1536 \
ROPE_DIMS=24 \
TTT_EVAL_ENABLED=0 \
COMPILE_ENABLED=1 \
COMPILE_FULLGRAPH=0 \
MODEL_PATH="${MODEL_PATH}" \
SWEEP_MAX_SECONDS="${SWEEP_MAX_SECONDS}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/sweep_ngram.py" \
    2>&1 | tee "logs/sweep_ngram_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "============================================"
echo "  SWEEP DONE — results in sweep_ngram_results.csv"
echo "============================================"
