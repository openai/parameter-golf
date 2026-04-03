#!/bin/bash
set -euo pipefail
# A/B TEST: SIPHON_ENABLED=0 (control)
# 200s wallclock for quick directional signal
# Uses siphon/train_gpt.py with siphon OFF to isolate the variable

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
COMPILE_ENABLED="${COMPILE_ENABLED:-1}"
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"
TORCHDYNAMO_SUPPRESS_ERRORS="${TORCHDYNAMO_SUPPRESS_ERRORS:-1}"

echo "============================================"
echo "  A/B TEST: SIPHON=OFF (control)"
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
WARMDOWN_ITERS=2000 \
SIPHON_ENABLED=0 \
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
COMPILE_ENABLED="${COMPILE_ENABLED}" \
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH}" \
TORCHDYNAMO_SUPPRESS_ERRORS="${TORCHDYNAMO_SUPPRESS_ERRORS}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${REPO_ROOT}/experiments/Rat_Rod/siphon/train_gpt.py" \
    2>&1 | tee "logs/ab_siphon_off_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE — SIPHON=OFF (control)"
echo "============================================"
