#!/bin/bash
# Cubric — Cadence sweep: how often should the bank fire?
# Single GPU by default. ~2 min per arm, ~8 min total.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
if [ -d "flash-attention/hopper" ]; then
    export PYTHONPATH="$(pwd)/flash-attention/hopper:${PYTHONPATH:-}"
fi

NPROC="${NPROC:-1}"
SCRIPT="concepts/cubric/train_cubric.py"
RESULTS="concepts/cubric/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Base config: fast small model
BASE_ENV="SEED=1337 \
  NUM_LAYERS=8 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3 MLP_MULT=3 \
  BIGRAM_VOCAB_SIZE=1024 BIGRAM_DIM=64 XSA_LAST_N=4 \
  VE_ENABLED=1 VE_DIM=64 VE_LAYERS=6,7 \
  ROPE_DIMS=16 LN_SCALE=1 \
  TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 \
  MAX_WALLCLOCK_SECONDS=150 WARMDOWN_ITERS=875 WARMUP_STEPS=10 \
  VAL_LOSS_EVERY=100 VAL_BATCH_SIZE=524288 \
  LATE_QAT_THRESHOLD=0.5 SWA_ENABLED=1 SWA_EVERY=50 QAT_ENABLED=0 \
  TTT_EVAL_ENABLED=0 \
  CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2"

echo "═══════════════════════════════════════════"
echo "  CUBRIC — Cadence Sweep (8L/384d)"
echo "  NPROC=$NPROC"
echo "═══════════════════════════════════════════"

run_arm() {
    local cadence=$1
    local run_id="cubric_cad${cadence}_${TIMESTAMP}"
    mkdir -p "$RESULTS/${run_id}"
    echo ""
    echo "── cadence=$cadence ──"
    env $BASE_ENV \
        RUN_ID="$run_id" \
        CRAWLER_BANK_CADENCE="$cadence" \
        torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT" 2>&1 \
        | tee "$RESULTS/${run_id}/log.txt"
    cp final_model.pt "$RESULTS/${run_id}/final.pt" 2>/dev/null || true
    echo "done: $run_id"
}

# Control: no bank
echo ""
echo "── control (no bank) ──"
CTRL_ID="cubric_ctrl_${TIMESTAMP}"
mkdir -p "$RESULTS/${CTRL_ID}"
env $BASE_ENV \
    RUN_ID="$CTRL_ID" \
    CRAWLER_BANK_ENABLED=0 \
    CRAWLER_BANK_CADENCE=1 \
    torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT" 2>&1 \
    | tee "$RESULTS/${CTRL_ID}/log.txt"
cp final_model.pt "$RESULTS/${CTRL_ID}/final.pt" 2>/dev/null || true
echo "done: $CTRL_ID"

# Cadence arms
run_arm 1    # every step (baseline bank, max overhead)
run_arm 4    # every 4th step
run_arm 10   # every 10th step (skiptrace sweet spot?)
run_arm 20   # every 20th step

echo ""
echo "═══════════════════════════════════════════"
echo "  CUBRIC SWEEP COMPLETE"
echo "  Results: $RESULTS/"
echo "═══════════════════════════════════════════"
