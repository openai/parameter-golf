#!/bin/bash
set -euo pipefail
# ══════════════════════════════════════════════════════════════
# CUBRIC CADENCE ACCUMULATOR — N/N/N/C pattern
#
# HYPOTHESIS: Periodic neural optimization of n-gram hash tables
# will improve BPP over static tables. The C-step uses already-scored
# data to: (1) decay stale counts, (2) boost patterns where model and
# n-gram agree, (3) prune noisy hash collisions, (4) reweight orders
# by tracked accuracy. This transforms the n-gram system from a
# static counter into an adaptive pattern reservoir.
#
# EXPECTED: 0.003-0.010 BPP improvement over baseline n-gram.
# The improvement should grow over the eval pass as the C-step
# accumulates more signal about the document.
#
# RISK: C-step could corrupt the tables if pruning/boosting is
# miscalibrated. Count decay could erase good patterns.
#
# ARMS:
#   A: Baseline (n-gram, no cubric)
#   B: Cubric cadence=4 (C every 4 batches, frequent optimization)
#   C: Cubric cadence=10 (C every 10 batches, balanced)
#   D: Cubric cadence=20 (C every 20 batches, conservative)
#
# Score-first legal: C-step only reads from already-scored segments.
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
NPROC="${NPROC_PER_NODE:-8}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt_cadence.py"

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
    local run_id="cubcad_${arm_id}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
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
    echo "── [${arm_id}] result ──"
    grep -E "final_int6_sliding_window_ngram.*exact|c_steps=" \
      "logs/${run_id}.log" 2>/dev/null | tail -3
    echo ""
}

mkdir -p logs

echo "══════════════════════════════════════════════════"
echo "  CUBRIC CADENCE — N/N/N/C ACCUMULATOR A/B"
echo "══════════════════════════════════════════════════"

run_arm "A" "CONTROL: static n-gram, no cubric" \
  CUBRIC_CADENCE=0

run_arm "B" "H: C every 4 batches (aggressive optimization)" \
  CUBRIC_CADENCE=4 CUBRIC_COUNT_DECAY=0.02 \
  CUBRIC_BOOST_CONFIDENT=1 CUBRIC_PRUNE_NOISY=1 CUBRIC_REWEIGHT_ORDERS=1

run_arm "C" "H: C every 10 batches (balanced)" \
  CUBRIC_CADENCE=10 CUBRIC_COUNT_DECAY=0.02 \
  CUBRIC_BOOST_CONFIDENT=1 CUBRIC_PRUNE_NOISY=1 CUBRIC_REWEIGHT_ORDERS=1

run_arm "D" "H: C every 20 batches (conservative)" \
  CUBRIC_CADENCE=20 CUBRIC_COUNT_DECAY=0.02 \
  CUBRIC_BOOST_CONFIDENT=1 CUBRIC_PRUNE_NOISY=1 CUBRIC_REWEIGHT_ORDERS=1

echo "══════════════════════════════════════════════════"
echo "  SUMMARY"
echo "══════════════════════════════════════════════════"
for f in logs/cubcad_*_s${SEED}_*.log; do
    arm=$(basename "$f" | sed 's/cubcad_\([A-D]\)_.*/\1/')
    bpb=$(grep "final_int6_sliding_window_ngram.*exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    csteps=$(grep -oP 'c_steps=\K[0-9]+' "$f" 2>/dev/null | tail -1 || echo "0")
    echo "  [$arm] sliding_ngram_bpb=$bpb c_steps=$csteps"
done
echo "══════════════════════════════════════════════════"
