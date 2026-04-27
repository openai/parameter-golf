#!/bin/bash
set -euo pipefail
# BASELINE runner for master_run.sh — green v1 config with overridable wallclock
# Usage: MAX_WALLCLOCK_SECONDS=180 NPROC_PER_NODE=8 bash experiments/baseline_run.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PATH="/home/frosty40/miniconda3/bin:${PATH}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}" \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=1 \
LATE_QAT_THRESHOLD=0 \
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
    2>&1 | tee "logs/baseline_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
