#!/bin/bash
# Cubric — Full evaluation: all axes
# Single GPU by default. ~30 min total.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
if [ -d "flash-attention/hopper" ]; then
    export PYTHONPATH="$(pwd)/flash-attention/hopper:${PYTHONPATH:-}"
fi

NPROC="${NPROC:-1}"
SCRIPT="concepts/cubric/train_cubric.py"
RESULTS="concepts/cubric/results"
TS=$(date +%Y%m%d_%H%M%S)

BASE="SEED=1337 \
  TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 \
  MAX_WALLCLOCK_SECONDS=150 WARMDOWN_ITERS=875 WARMUP_STEPS=10 \
  VAL_LOSS_EVERY=100 VAL_BATCH_SIZE=524288 \
  LATE_QAT_THRESHOLD=0.5 SWA_ENABLED=1 SWA_EVERY=50 QAT_ENABLED=0 \
  TTT_EVAL_ENABLED=0 ROPE_DIMS=16 LN_SCALE=1"

run() {
    local label=$1; shift
    local run_id="cubric_${label}_${TS}"
    mkdir -p "$RESULTS/${run_id}"
    echo ""
    echo "══ $label ══"
    env $BASE "$@" RUN_ID="$run_id" \
        torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT" 2>&1 \
        | tee "$RESULTS/${run_id}/log.txt"
    cp final_model.pt "$RESULTS/${run_id}/final.pt" 2>/dev/null || true
}

echo "═══════════════════════════════════════════"
echo "  CUBRIC — Full Evaluation"
echo "═══════════════════════════════════════════"

# ── Axis 1: Cadence (8L/384d) ──
SMALL="NUM_LAYERS=8 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3 MLP_MULT=3 \
  BIGRAM_VOCAB_SIZE=1024 BIGRAM_DIM=64 XSA_LAST_N=4 \
  VE_ENABLED=1 VE_DIM=64 VE_LAYERS=6,7"

run "ax1_ctrl"  $SMALL CRAWLER_BANK_ENABLED=0 CRAWLER_BANK_CADENCE=1
run "ax1_cad1"  $SMALL CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 CRAWLER_BANK_CADENCE=1
run "ax1_cad4"  $SMALL CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 CRAWLER_BANK_CADENCE=4
run "ax1_cad10" $SMALL CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 CRAWLER_BANK_CADENCE=10
run "ax1_cad20" $SMALL CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 CRAWLER_BANK_CADENCE=20
run "ax1_cad50" $SMALL CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 CRAWLER_BANK_CADENCE=50

# ── Axis 4: Bank Depth (8L/384d, cadence 10) ──
run "ax4_loop1" $SMALL CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=1 CRAWLER_BANK_CADENCE=10
run "ax4_loop2" $SMALL CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 CRAWLER_BANK_CADENCE=10
run "ax4_loop3" $SMALL CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=3 CRAWLER_BANK_CADENCE=10

# ── Axis 5: Model Scale (cadence 10, loops 2) ──
TINY="NUM_LAYERS=6 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=3 \
  BIGRAM_VOCAB_SIZE=512 BIGRAM_DIM=32 XSA_LAST_N=3 \
  VE_ENABLED=1 VE_DIM=32 VE_LAYERS=4,5"

MED="NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
  BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 XSA_LAST_N=4 \
  VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10"

run "ax5_tiny_ctrl"  $TINY CRAWLER_BANK_ENABLED=0 CRAWLER_BANK_CADENCE=1
run "ax5_tiny_skip"  $TINY CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 CRAWLER_BANK_CADENCE=10
run "ax5_small_ctrl" $SMALL CRAWLER_BANK_ENABLED=0 CRAWLER_BANK_CADENCE=1
run "ax5_small_skip" $SMALL CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 CRAWLER_BANK_CADENCE=10
run "ax5_med_ctrl"   $MED CRAWLER_BANK_ENABLED=0 CRAWLER_BANK_CADENCE=1
run "ax5_med_skip"   $MED CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 CRAWLER_BANK_CADENCE=10

echo ""
echo "═══════════════════════════════════════════"
echo "  CUBRIC EVAL COMPLETE — $(ls -d $RESULTS/cubric_*_$TS 2>/dev/null | wc -l) runs"
echo "  Results: $RESULTS/"
echo "═══════════════════════════════════════════"
