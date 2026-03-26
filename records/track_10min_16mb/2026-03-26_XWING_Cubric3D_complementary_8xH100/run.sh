#!/bin/bash
set -euo pipefail
# X-WING YELLOW III: Yellow II + warm-start cubric
# Warm-start: initialize multipliers at proven converged values, not 1.0
# Full power from chunk 1 instead of wasting 30 chunks converging

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-2045}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "============================================"
echo "  X-WING YELLOW II — THE MONSTER"
echo "  Seed: ${SEED}"
echo "  3D cubric: order × entropy × count (54 mults)"
echo "  Complementary training: alpha=0.5"
echo "  Eval alpha: 0.20-0.75 | Orders: 2-9"
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
VAL_LOSS_EVERY=20000 \
TRAIN_LOG_EVERY=1000 \
SWA_EVERY=100 \
COMPLEMENT_ALPHA=0.5 \
NGRAM_EVAL_ORDER=9 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MIN=0.20 \
NGRAM_EVAL_ALPHA_MAX=0.75 \
NGRAM_EVAL_ENTROPY_CENTER=3.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MIN_COUNT=2 \
NGRAM_EVAL_BUCKETS=8388608 \
NGRAM_EVAL_MAX_SECONDS=300 \
CUBRIC_CADENCE="${CUBRIC_CADENCE:-32}" \
COMPILE_FULLGRAPH=0 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/xwing_yellow2_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
