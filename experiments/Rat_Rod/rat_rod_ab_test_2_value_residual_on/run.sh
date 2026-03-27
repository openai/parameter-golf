#!/bin/bash
set -euo pipefail
# A/B TEST: VALUE_RESIDUAL=1 (test)
# 200s wallclock for quick directional signal

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "============================================"
echo "  A/B TEST: VALUE_RESIDUAL=1 (test)"
echo "  Seed: ${SEED}  |  200s wallclock"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=200 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
LATE_QAT_THRESHOLD=0 \
VALUE_RESIDUAL=1 \
NGRAM_EVAL_ORDER=9 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MIN=0.05 \
NGRAM_EVAL_ALPHA_MAX=0.60 \
NGRAM_EVAL_ENTROPY_CENTER=3.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MIN_COUNT=2 \
NGRAM_EVAL_BUCKETS=8388608 \
NGRAM_EVAL_MAX_SECONDS=0 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=1 \
NGRAM_ORDER_MULTS="0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${REPO_ROOT}/experiments/Rat_Rod/green/train_gpt.py" \
    2>&1 | tee "logs/ab_vr_on_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE — VALUE_RESIDUAL=1"
echo "============================================"
