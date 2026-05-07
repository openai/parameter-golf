#!/bin/bash
set -euo pipefail
# N-gram parameter sweep — find optimal settings for the 0.96 regime
# Each arm changes ONE variable from the baseline.
# Single GPU, COMPILE_ENABLED=0 for Vast compat.
#
# HYPOTHESES:
# 1. Higher n-gram order (8,9) captures longer patterns the 7-gram misses
# 2. More buckets (8M,16M) reduces collisions — cleaner data = better blend
# 3. Min count 1 catches more patterns at cost of noise
# 4. Alpha range may be suboptimal — the 0.96 model is more confident
# 5. Entropy center/scale tuned for 1.12 model, not 0.96

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/local_shims:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC=1
SCRIPT="${SCRIPT_DIR}/train_gpt.py"

# Baseline settings (from PR #753)
BASE=(
  SEED="${SEED}" COMPILE_ENABLED=0
  MLP_ACT=leaky_relu_sq MLP_LEAKY_SLOPE=0.5
  XSA_LAST_N=4 BIGRAM_VOCAB_SIZE=1536 ROPE_DIMS=24
  NGRAM_EVAL_ADAPTIVE=1
  NGRAM_EVAL_ALPHA=0.30 NGRAM_EVAL_ALPHA_MIN=0.05 NGRAM_EVAL_ALPHA_MAX=0.60
  NGRAM_EVAL_ENTROPY_CENTER=4.0 NGRAM_EVAL_ENTROPY_SCALE=2.0
  NGRAM_EVAL_MIN_COUNT=2 NGRAM_EVAL_BUCKETS=4194304
  NGRAM_EVAL_ORDER=7
)

run_arm() {
    local arm_id="$1"; local hyp="$2"; shift 2
    local run_id="sweep_${arm_id}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
    echo ""
    echo "═══════════════════════════════════════"
    echo "  [${arm_id}] ${hyp}"
    echo "  RUN_ID: ${run_id}"
    echo "═══════════════════════════════════════"
    env "${BASE[@]}" "$@" RUN_ID="$run_id" \
      torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT" \
      2>&1 | tee "logs/${run_id}.log"
    echo "── [${arm_id}] ──"
    grep -E "final_int6_sliding_window_ngram.*exact" "logs/${run_id}.log" 2>/dev/null | tail -1
    echo ""
}

mkdir -p logs
echo "══════════════════════════════════════════"
echo "  N-GRAM PARAMETER SWEEP (1-GPU)"
echo "══════════════════════════════════════════"

# ── Order sweep ──
run_arm "ord8" "H: 8-gram captures longer patterns" NGRAM_EVAL_ORDER=8
run_arm "ord9" "H: 9-gram even longer context" NGRAM_EVAL_ORDER=9

# ── Bucket sweep ──
run_arm "bkt8M" "H: 8M buckets = fewer collisions" NGRAM_EVAL_BUCKETS=8388608
run_arm "bkt16M" "H: 16M buckets = minimal collisions" NGRAM_EVAL_BUCKETS=16777216

# ── Min count ──
run_arm "mc1" "H: min_count=1 catches more patterns" NGRAM_EVAL_MIN_COUNT=1
run_arm "mc3" "H: min_count=3 cleaner matches" NGRAM_EVAL_MIN_COUNT=3

# ── Alpha range ──
run_arm "alpha_tight" "H: tighter alpha for confident model" NGRAM_EVAL_ALPHA_MIN=0.10 NGRAM_EVAL_ALPHA_MAX=0.45
run_arm "alpha_wide" "H: wider alpha for aggressive blend" NGRAM_EVAL_ALPHA_MIN=0.02 NGRAM_EVAL_ALPHA_MAX=0.75

# ── Entropy sigmoid ──
run_arm "ent_low" "H: lower entropy center (model is more confident at 0.96)" NGRAM_EVAL_ENTROPY_CENTER=3.0
run_arm "ent_steep" "H: steeper sigmoid = sharper alpha transitions" NGRAM_EVAL_ENTROPY_SCALE=3.5

echo "══════════════════════════════════════════"
echo "  SUMMARY"
echo "══════════════════════════════════════════"
for f in logs/sweep_*_s${SEED}_*.log; do
    arm=$(basename "$f" | sed "s/sweep_\(.*\)_s${SEED}.*/\1/")
    bpb=$(grep "final_int6_sliding_window_ngram.*exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    echo "  [$arm] = $bpb"
done
echo "══════════════════════════════════════════"
