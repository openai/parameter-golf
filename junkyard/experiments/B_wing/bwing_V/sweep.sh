#!/bin/bash
set -euo pipefail
# N-gram parameter grid sweep on saved bwing_V model
# Loads final_model.int6.ptz once, runs ~192 eval configs (~3 min each)
# Results: experiments/B_wing/bwing_V/sweep_results.csv

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "============================================"
echo "  B-WING V — N-gram Parameter Sweep"
echo "  Model: final_model.int6.ptz (from bwing_V run)"
echo "  Grid: alpha_max × entropy_center × high_order_mult × min_count × cubric"
echo "============================================"

# Base env vars for model architecture (must match training)
SEED=1337 \
F1_CORR_RANK=0 \
MLP_ACT=leaky_relu_sq \
MLP_LEAKY_SLOPE=0.5 \
XSA_LAST_N=4 \
BIGRAM_VOCAB_SIZE=1536 \
ROPE_DIMS=24 \
NGRAM_EVAL_ORDER=9 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_BUCKETS=8388608 \
NGRAM_ENTROPY_SHIFT=1 \
COMPILE_FULLGRAPH=0 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/eval_sweep.py" \
    2>&1 | tee "${SCRIPT_DIR}/sweep_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  SWEEP DONE — check sweep_results.csv"
echo "============================================"
