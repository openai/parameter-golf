#!/bin/bash
set -euo pipefail
# Podracer GREEN: SOTA + cubric lite (per-order adaptive alpha)
# Base: verified SOTA 147bbccc + cubric lite overlay
# CUBRIC_CADENCE=0 → identical to SOTA. >0 → per-order alpha scaling.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-2045}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "============================================"
echo "  PODRACER GREEN (cubric lite)"
echo "  Seed: ${SEED}"
echo "  Cubric cadence: ${CUBRIC_CADENCE:-32}"
echo "============================================"

SEED="$SEED" \
F1_CORR_RANK=0 \
DISTILL_ENABLED=0 \
MLP_ACT=leaky_relu_sq \
MLP_LEAKY_SLOPE=0.5 \
XSA_LAST_N=4 \
BIGRAM_VOCAB_SIZE=1536 \
TTT_EVAL_ENABLED=0 \
ROPE_DIMS=24 \
NGRAM_EVAL_ORDER=7 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MIN=0.05 \
NGRAM_EVAL_ALPHA_MAX=0.60 \
NGRAM_EVAL_ENTROPY_CENTER=4.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MIN_COUNT=2 \
NGRAM_EVAL_BUCKETS=4194304 \
NGRAM_EVAL_MAX_SECONDS=300 \
CUBRIC_CADENCE="${CUBRIC_CADENCE:-32}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/podracer_green_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
