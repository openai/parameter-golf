#!/bin/bash
set -euo pipefail
# Cubric n-gram accumulator — FULL A/B with all variants
# Uses the evalonly script (no training changes)
#
# Arms:
#   A: Baseline (entropy-adaptive alpha, no cubric)
#   B: Cubric basic (alpha bounds shift from reliability)
#   C: Cubric + per-order (backoff reranked by per-order reliability)
#   D: Cubric + agreement weighting (boost alpha when model & ngram agree)
#   E: Cubric + entropy adaptation (sigmoid shifts to match document)
#   F: Cubric ALL (B+C+D+E combined)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
NPROC="${NPROC_PER_NODE:-8}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt_evalonly.py"

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

run_arm() {
    local arm_id="$1"
    local desc="$2"
    shift 2
    local run_id="cubric_${arm_id}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
    echo ""
    echo "═══════════════════════════════════════"
    echo "  [${arm_id}] ${desc}"
    echo "  RUN_ID: ${run_id}"
    echo "═══════════════════════════════════════"
    env "${COMMON_ENV[@]}" "$@" \
      RUN_ID="$run_id" \
      torchrun --standalone --nproc_per_node="$NPROC" \
        "$TRAIN_SCRIPT" \
        2>&1 | tee "logs/${run_id}.log"
    echo "── [${arm_id}] summary ──"
    grep -E "final_int6_sliding_window_ngram.*exact|cubric_rel=|ngram_eval:cutoff" "logs/${run_id}.log" 2>/dev/null || true
    echo ""
}

echo "═══════════════════════════════════════════════"
echo "  CUBRIC N-GRAM — FULL A/B SWEEP"
echo "═══════════════════════════════════════════════"

run_arm "A" "Baseline (no cubric)" \
  CUBRIC_ENABLED=0

run_arm "B" "Cubric basic (alpha bounds shift)" \
  CUBRIC_ENABLED=1 CUBRIC_DECAY=0.95 CUBRIC_BOOST_SCALE=0.15

run_arm "C" "Cubric + per-order reliability" \
  CUBRIC_ENABLED=1 CUBRIC_DECAY=0.95 CUBRIC_BOOST_SCALE=0.15 \
  CUBRIC_PER_ORDER=1

run_arm "D" "Cubric + agreement weighting" \
  CUBRIC_ENABLED=1 CUBRIC_DECAY=0.95 CUBRIC_BOOST_SCALE=0.15 \
  CUBRIC_AGREEMENT=1 CUBRIC_AGREEMENT_SCALE=2.0

run_arm "E" "Cubric + entropy sigmoid adaptation" \
  CUBRIC_ENABLED=1 CUBRIC_DECAY=0.95 CUBRIC_BOOST_SCALE=0.15 \
  CUBRIC_ENTROPY_ADAPT=1

run_arm "F" "Cubric ALL (B+C+D+E)" \
  CUBRIC_ENABLED=1 CUBRIC_DECAY=0.95 CUBRIC_BOOST_SCALE=0.15 \
  CUBRIC_PER_ORDER=1 CUBRIC_AGREEMENT=1 CUBRIC_AGREEMENT_SCALE=2.0 \
  CUBRIC_ENTROPY_ADAPT=1

echo "═══════════════════════════════════════════════"
echo "  ALL ARMS COMPLETE — compare logs/"
echo "═══════════════════════════════════════════════"
grep -h "final_int6_sliding_window_ngram.*exact" logs/cubric_*_s${SEED}_*.log 2>/dev/null || true
