#!/usr/bin/env bash
set -euo pipefail

# Exact reproduction of the 1.1303 baseline result
# Uses sota254/train_gpt.py with original settings from README
# Purpose: verify baseline reproduces on this pod with current FA3 build

LOGDIR="logs/baseline_repro_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  Baseline Reproduction (target: 1.1303)"
echo "  Code: sota254/train_gpt.py"
echo "  Logs: $LOGDIR"
echo "============================================"

SEED="${SEED:-1337}" \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_MOMENTUM=0.9 \
NCCL_IB_DISABLE=1 \
RUN_ID="baseline_repro_s${SEED:-1337}" \
torchrun --standalone --nproc_per_node="${NPROC:-8}" \
    sota254/train_gpt.py \
    2>&1 | tee "$LOGDIR/run_s${SEED:-1337}.log"

echo ""
echo "  Target: 1.1303 BPB (sliding), 1.1528 (roundtrip)"
f="$LOGDIR/run_s${SEED:-1337}.log"
for label in int6_roundtrip int6_sliding_window; do
    bpb=$(grep -oP "final_${label}\S* val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: $bpb" || true
done
