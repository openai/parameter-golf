#!/bin/bash
set -euo pipefail
# Cubric cadence C only — balanced cadence=10
# HYPOTHESIS: C every 10 batches is the sweet spot — enough data per C-step
# to make good decisions, low enough overhead to not slow eval.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -d "flash-attention/hopper" ]; then
    export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
elif [ -d "local_shims" ]; then
    export PYTHONPATH="${REPO_ROOT}/local_shims:${PYTHONPATH:-}"
fi

SEED="${SEED:-1337}"
NPROC="${NPROC_PER_NODE:-8}"
RUN_ID="cubcad_C_cad10_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "═══════════════════════════════════════"
echo "  CUBRIC CADENCE=10 (balanced)"
echo "  RUN_ID: ${RUN_ID}"
echo "═══════════════════════════════════════"

env \
  SEED="${SEED}" \
  RUN_ID="${RUN_ID}" \
  MLP_ACT=leaky_relu_sq MLP_LEAKY_SLOPE=0.5 \
  XSA_LAST_N=4 BIGRAM_VOCAB_SIZE=1536 \
  ROPE_DIMS=24 \
  COMPILE_ENABLED=1 COMPILE_FULLGRAPH=0 \
  NGRAM_EVAL_ORDER=5 \
  NGRAM_EVAL_ALPHA=0.30 \
  NGRAM_EVAL_MIN_COUNT=2 \
  NGRAM_EVAL_BUCKETS=4194304 \
  NGRAM_EVAL_ADAPTIVE=1 \
  NGRAM_EVAL_ALPHA_MIN=0.05 \
  NGRAM_EVAL_ALPHA_MAX=0.60 \
  CUBRIC_CADENCE=10 \
  CUBRIC_COUNT_DECAY=0.02 \
  CUBRIC_BOOST_CONFIDENT=1 \
  CUBRIC_PRUNE_NOISY=1 \
  CUBRIC_REWEIGHT_ORDERS=1 \
  torchrun --standalone --nproc_per_node="$NPROC" \
    "${SCRIPT_DIR}/train_gpt_cadence.py" \
    2>&1 | tee "logs/${RUN_ID}.log"

echo ""
echo "── RESULT ──"
grep -E "final_int6_sliding_window_ngram.*exact|final_int6_sliding_window_exact|c_steps=" \
  "logs/${RUN_ID}.log" 2>/dev/null | tail -5
echo "═══════════════════════════════════════"
