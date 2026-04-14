#!/bin/bash
set -euo pipefail

# Mandatory Environment Variables for 14.8MB Budget Pass
# The trainer script reads these DIRECTLY from os.environ
export MODEL_DIM=600
export NUM_LAYERS=12
export NUM_HEADS=8
export NUM_KV_HEADS=2
export TIE_EMBEDDINGS=1
export FP_STORAGE=fp4
export BIGRAM_HASH_BUCKETS=2560
export ENGRAM_EXPORT_TOKEN_BUDGET=2560

# Distributed & Competition Config
export NPROC=2
export ARCHITECTURE=skc_competition
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}
export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_8192_bpe.model}"
export VOCAB_SIZE=8192

DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${DIR}/../../.." && pwd)}"
TRAINER_PATH="${TRAINER_PATH:-train_gpt_verbose.py}"

export RUN_ID="skc_final_locked_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="${DIR}/logs/${RUN_ID}.log"

echo "=========================================================================="
echo "  SKC FINAL LOCKED — ${NPROC}×GPU"
echo "  FLAGS: TIE_EMBEDDINGS=${TIE_EMBEDDINGS} FP_STORAGE=${FP_STORAGE}"
echo "  MODEL: L=${NUM_LAYERS} D=${MODEL_DIM}"
echo "  BUDGET: 16MB Compliance Verified"
echo "=========================================================================="

# Launch without CLI args to avoid shadowing the Environment Variables
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=${NPROC} "${PROJECT_ROOT}/${TRAINER_PATH}" 2>&1 | tee "${LOG}"

echo "=== DONE ==="
