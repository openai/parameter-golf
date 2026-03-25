#!/bin/bash
set -euo pipefail
# Podracer RED: safety-first lane
# - hard-disables TTT (no eval-time gradient updates)
# - keeps legal score-first n-gram backoff
# - applies conservative n-gram stability/coverage upgrades

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-2045}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Base Podracer II architecture (kept fixed for apples-to-apples legality)
export F1_CORR_RANK="${F1_CORR_RANK:-0}"
export DISTILL_ENABLED="${DISTILL_ENABLED:-0}"
export MLP_ACT="${MLP_ACT:-leaky_relu_sq}"
export MLP_LEAKY_SLOPE="${MLP_LEAKY_SLOPE:-0.5}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}"
export ROPE_DIMS="${ROPE_DIMS:-24}"

# Hard safety lock: no test-time training path
export TTT_EVAL_ENABLED=0
export TTT_EPOCHS=0
export TTT_MAX_TRAIN_CHUNKS=0

# Conservative n-gram improvements (safe: no target-leak oracle/min-loss routing)
export NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-7}"
export NGRAM_EVAL_MIN_ORDER="${NGRAM_EVAL_MIN_ORDER:-2}"
export NGRAM_EVAL_ADAPTIVE="${NGRAM_EVAL_ADAPTIVE:-1}"
export NGRAM_EVAL_ALPHA="${NGRAM_EVAL_ALPHA:-0.30}"
export NGRAM_EVAL_ALPHA_MIN="${NGRAM_EVAL_ALPHA_MIN:-0.05}"
export NGRAM_EVAL_ALPHA_MAX="${NGRAM_EVAL_ALPHA_MAX:-0.70}"
export NGRAM_EVAL_ENTROPY_CENTER="${NGRAM_EVAL_ENTROPY_CENTER:-3.0}"
export NGRAM_EVAL_ENTROPY_SCALE="${NGRAM_EVAL_ENTROPY_SCALE:-2.0}"
export NGRAM_EVAL_MIN_COUNT="${NGRAM_EVAL_MIN_COUNT:-2}"
export NGRAM_EVAL_BUCKETS="${NGRAM_EVAL_BUCKETS:-8388608}"
export NGRAM_EVAL_MAX_SECONDS="${NGRAM_EVAL_MAX_SECONDS:-300}"

echo "============================================"
echo "  PODRACER RED (safe lane)"
echo "  Seed: ${SEED}"
echo "  TTT: disabled"
echo "  NGRAM: order=${NGRAM_EVAL_ORDER} alpha_max=${NGRAM_EVAL_ALPHA_MAX} center=${NGRAM_EVAL_ENTROPY_CENTER} buckets=${NGRAM_EVAL_BUCKETS}"
echo "============================================"

SEED="${SEED}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/podracer_red_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
