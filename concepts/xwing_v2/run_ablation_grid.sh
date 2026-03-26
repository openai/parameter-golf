#!/bin/bash
set -euo pipefail
# 2x2 ablation grid: cubric × per-order entropy centers
# Loads existing checkpoint, eval only (~4 min each, ~16 min total)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

COMMON="NGRAM_EVAL_ORDER=7 NGRAM_EVAL_ALPHA_MAX=0.70 NGRAM_EVAL_ENTROPY_CENTER=3.0 NGRAM_EVAL_ENTROPY_SCALE=2.0 NGRAM_EVAL_BUCKETS=8388608 NGRAM_EVAL_MIN_COUNT=2 NGRAM_EVAL_ALPHA_MIN=0.05 NGRAM_EVAL_ALPHA=0.30 COMPILE_FULLGRAPH=0 MLP_ACT=leaky_relu_sq MLP_LEAKY_SLOPE=0.5 XSA_LAST_N=4 BIGRAM_VOCAB_SIZE=1536 ROPE_DIMS=24"
NPROC="${NPROC_PER_NODE:-8}"

echo "============================================"
echo "  2x2 ABLATION GRID"
echo "  cubric × per-order entropy centers"
echo "============================================"
echo ""

echo ">>> [1/4] Flat baseline (no cubric, no per-order)"
env $COMMON CUBRIC_CADENCE=0 PER_ORDER_ENT=0 \
  torchrun --standalone --nproc_per_node="$NPROC" "${SCRIPT_DIR}/eval_only.py" 2>&1 | grep -E "RESULT|eval_only:"
echo ""

echo ">>> [2/4] Cubric only (v1 equivalent)"
env $COMMON CUBRIC_CADENCE=1 PER_ORDER_ENT=0 \
  torchrun --standalone --nproc_per_node="$NPROC" "${SCRIPT_DIR}/eval_only.py" 2>&1 | grep -E "RESULT|eval_only:"
echo ""

echo ">>> [3/4] Per-order centers only"
env $COMMON CUBRIC_CADENCE=0 PER_ORDER_ENT=1 \
  torchrun --standalone --nproc_per_node="$NPROC" "${SCRIPT_DIR}/eval_only.py" 2>&1 | grep -E "RESULT|eval_only:"
echo ""

echo ">>> [4/4] Both (v2 full)"
env $COMMON CUBRIC_CADENCE=1 PER_ORDER_ENT=1 \
  torchrun --standalone --nproc_per_node="$NPROC" "${SCRIPT_DIR}/eval_only.py" 2>&1 | grep -E "RESULT|eval_only:"
echo ""

echo "============================================"
echo "  GRID COMPLETE"
echo "============================================"
