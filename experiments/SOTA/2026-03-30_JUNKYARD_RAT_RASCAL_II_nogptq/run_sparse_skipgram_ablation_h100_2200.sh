#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

SEED="${SEED:-444}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

mkdir -p logs

echo "============================================"
echo "  RASCAL SPARSE-SKIPGRAM ABLATION"
echo "  Seed: ${SEED}"
echo "  nproc: ${NPROC_PER_NODE}"
echo "  Iterations: 2200"
echo "  Warmdown: 0"
echo "============================================"

SEED="${SEED}" \
ITERATIONS=2200 \
WARMDOWN_ITERS=0 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1100 \
POST_EMA_DIAGNOSTIC=1 \
SKIP_FINAL_EVAL=0 \
NGRAM_EVAL_ORDER=7 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_MAX_SECONDS=180 \
NGRAM_SPARSE_PATTERNS="1,3;1,2;1,3,5;1,2,4;1,3,5,7;1,2,4,8;1,3,5,7,9;1,2,4,8,16;1,3,5,7,9,11;1,2,4,8,16,32" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${SCRIPT_DIR}/train_gpt_sparse_skipgram_ablation.py" \
  2>&1 | tee "logs/rascal_sparse_skipgram_ablation_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
