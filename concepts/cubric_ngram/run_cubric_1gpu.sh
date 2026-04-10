#!/bin/bash
set -euo pipefail
# Cubric n-gram eval-only A/B — single GPU (Vast.ai $1/hr)
# Each arm trains the SAME model, only eval-time n-gram blending differs.
# All arms share one training run, then eval 5 ways.
#
# ══════════════════════════════════════════════════════════════
# HYPOTHESES
# ══════════════════════════════════════════════════════════════
#
# ARM A (control): Entropy-adaptive alpha with fixed bounds is already
#   near-optimal. Baseline for comparison.
#
# ARM B (basic cubric): Documents vary in n-gram predictability. Shifting
#   alpha bounds based on accumulated reliability will improve BPB on
#   heterogeneous eval data by ~0.001-0.003 vs fixed bounds.
#   RISK: The entropy formula already captures most of this signal.
#
# ARM C (per-order): Different documents favor different n-gram orders.
#   Code is trigram-heavy, prose is 5-gram-heavy. Per-order reliability
#   tracking that reranks backoff preference will improve BPP by 0.002-0.005.
#   RISK: With only 2 min_count, higher orders are sparse and noisy.
#
# ARM D (agreement): When model and n-gram both assign high probability
#   to the same token, the blend should be aggressive. Agreement weighting
#   captures a signal entropy alone misses: model confidence + n-gram
#   confidence = strong evidence. Expected: 0.001-0.003.
#   RISK: agreement_scale is hand-tuned, may overshoot.
#
# ARM E (entropy adapt): The sigmoid mapping assumes fixed entropy
#   distribution. Real documents have different entropy profiles (code ~2-3
#   bits, prose ~4-6 bits). Shifting the sigmoid to match running entropy
#   stats will improve calibration. Expected: 0.001-0.002.
#   RISK: Smallest expected effect. May be noise-level.
#
# ARM F (all combined): If mechanisms are orthogonal, gains should stack.
#   Expected: sum of individual gains minus some overlap (~60-80% of sum).
#   RISK: Interactions could cancel. More complexity = more noise.
#
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -d "flash-attention/hopper" ]; then
    export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
elif [ -d "local_shims" ]; then
    export PYTHONPATH="${REPO_ROOT}/local_shims:${PYTHONPATH:-}"
fi

SEED="${SEED:-1337}"
NPROC="${NPROC_PER_NODE:-1}"
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
    local hyp="$2"
    shift 2
    local run_id="cubric_${arm_id}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
    echo ""
    echo "═══════════════════════════════════════"
    echo "  [${arm_id}] ${hyp}"
    echo "  RUN_ID: ${run_id}"
    echo "═══════════════════════════════════════"
    env "${COMMON_ENV[@]}" "$@" \
      RUN_ID="$run_id" \
      torchrun --standalone --nproc_per_node="$NPROC" \
        "$TRAIN_SCRIPT" \
        2>&1 | tee "logs/${run_id}.log"
    echo "── [${arm_id}] result ──"
    grep -E "final_int6_sliding_window_ngram.*exact|final_int6_sliding_window_exact|cubric_rel=" \
      "logs/${run_id}.log" 2>/dev/null | tail -3
    echo ""
}

mkdir -p logs

echo "══════════════════════════════════════════════════"
echo "  CUBRIC N-GRAM — 1-GPU A/B (eval-only variants)"
echo "  NPROC=${NPROC} SEED=${SEED}"
echo "══════════════════════════════════════════════════"

run_arm "A" "CONTROL: entropy-adaptive alpha, no cubric" \
  CUBRIC_ENABLED=0

run_arm "B" "H: alpha bounds shift from accumulated reliability" \
  CUBRIC_ENABLED=1 CUBRIC_DECAY=0.95 CUBRIC_BOOST_SCALE=0.15

run_arm "C" "H: per-order reliability reranks backoff preference" \
  CUBRIC_ENABLED=1 CUBRIC_DECAY=0.95 CUBRIC_BOOST_SCALE=0.15 \
  CUBRIC_PER_ORDER=1

run_arm "D" "H: agreement weighting boosts alpha when model+ngram agree" \
  CUBRIC_ENABLED=1 CUBRIC_DECAY=0.95 CUBRIC_BOOST_SCALE=0.15 \
  CUBRIC_AGREEMENT=1 CUBRIC_AGREEMENT_SCALE=2.0

run_arm "E" "H: entropy sigmoid adapts to document entropy profile" \
  CUBRIC_ENABLED=1 CUBRIC_DECAY=0.95 CUBRIC_BOOST_SCALE=0.15 \
  CUBRIC_ENTROPY_ADAPT=1

run_arm "F" "H: all mechanisms combined — gains should stack if orthogonal" \
  CUBRIC_ENABLED=1 CUBRIC_DECAY=0.95 CUBRIC_BOOST_SCALE=0.15 \
  CUBRIC_PER_ORDER=1 CUBRIC_AGREEMENT=1 CUBRIC_AGREEMENT_SCALE=2.0 \
  CUBRIC_ENTROPY_ADAPT=1

echo "══════════════════════════════════════════════════"
echo "  SUMMARY"
echo "══════════════════════════════════════════════════"
for f in logs/cubric_*_s${SEED}_*.log; do
    arm=$(basename "$f" | sed 's/cubric_\([A-F]\)_.*/\1/')
    bpb=$(grep "final_int6_sliding_window_ngram.*exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    echo "  [$arm] sliding_ngram_bpb = $bpb"
done
echo "══════════════════════════════════════════════════"
