#!/usr/bin/env bash
set -euo pipefail

# EXP A: Multi-Token Prediction (MTP)
# Same SOTA base but with MTP_NUM_HEADS=2 during training.
# MTP heads are excluded from export → zero artifact size cost.
# Hypothesis: auxiliary future-token prediction loss improves internal representations.

LOGDIR="logs/exp_a_mtp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  EXP A: MTP-2 heads on SOTA 254 base"
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
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
NCCL_IB_DISABLE=1 \
RUN_ID="exp_a_mtp_s${SEED:-1337}" \
torchrun --standalone --nproc_per_node="${NPROC:-8}" \
    exp_a/train_gpt.py \
    2>&1 | tee "$LOGDIR/run_s${SEED:-1337}.log"

echo ""
echo "============================================"
echo "  EXP A Complete."
echo "============================================"
f="$LOGDIR/run_s${SEED:-1337}.log"
for label in int6_roundtrip int6_sliding_window; do
    bpb=$(grep -oP "final_${label}\S* val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: $bpb" || true
done
