#!/bin/bash
set -euo pipefail
# Cubric n-gram accumulator A/B test
# A: baseline (n-gram with entropy-adaptive alpha, no cubric)
# B: cubric accumulator (online alpha adaptation from n-gram reliability)
#
# Uses car02 SOTA as base. Single variable: CUBRIC_ENABLED.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
NPROC="${NPROC_PER_NODE:-8}"

COMMON_ENV=(
  SEED="${SEED}"
  MLP_ACT=leaky_relu_sq MLP_LEAKY_SLOPE=0.5
  XSA_LAST_N=4 BIGRAM_VOCAB_SIZE=1536
  ROPE_DIMS=24
  COMPILE_ENABLED=1 COMPILE_FULLGRAPH=0
  NGRAM_EVAL_ORDER=5
  NGRAM_EVAL_ALPHA=0.30
  NGRAM_EVAL_MIN_COUNT=2
  NGRAM_EVAL_BUCKETS=4194304
  NGRAM_EVAL_ADAPTIVE=1
  NGRAM_EVAL_ALPHA_MIN=0.05
  NGRAM_EVAL_ALPHA_MAX=0.60
)

echo "═══════════════════════════════════════"
echo "  CUBRIC N-GRAM ACCUMULATOR A/B TEST"
echo "═══════════════════════════════════════"

# ── ARM A: Baseline (no cubric) ──
echo ""
echo "── [A] Baseline: n-gram + entropy-adaptive alpha ──"
RUN_A="cubric_ng_A_baseline_$(date +%Y%m%d_%H%M%S)"
env "${COMMON_ENV[@]}" \
  RUN_ID="$RUN_A" \
  CUBRIC_ENABLED=0 \
  torchrun --standalone --nproc_per_node="$NPROC" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/${RUN_A}.log"

echo ""
echo "── [B] Cubric accumulator: online alpha adaptation ──"
RUN_B="cubric_ng_B_accum_$(date +%Y%m%d_%H%M%S)"
env "${COMMON_ENV[@]}" \
  RUN_ID="$RUN_B" \
  CUBRIC_ENABLED=1 \
  CUBRIC_DECAY=0.95 \
  CUBRIC_BOOST_SCALE=0.15 \
  torchrun --standalone --nproc_per_node="$NPROC" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/${RUN_B}.log"

echo ""
echo "═══════════════════════════════════════"
echo "  COMPARE:"
echo "═══════════════════════════════════════"
grep -h "final_int6_sliding_window_ngram.*exact\|cubric_rel=" \
  "logs/${RUN_A}.log" "logs/${RUN_B}.log" 2>/dev/null || true
