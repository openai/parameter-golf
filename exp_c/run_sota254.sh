#!/usr/bin/env bash
set -euo pipefail

# EXACT CLONE of PR #254 — Current best pending SOTA (1.1313 BPB)
# 11L Int6 MLP3x + SmearGate + BigramHash + TTT SGD 3 epochs
# Just run it. No modifications.

LOGDIR="logs/sota254_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  PR #254 EXACT CLONE — 1.1313 BPB target"
echo "  11L + TTT + SmearGate + BigramHash"
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
RUN_ID="sota254_s${SEED:-1337}" \
torchrun --standalone --nproc_per_node="${NPROC:-8}" \
    sota254/train_gpt.py \
    2>&1 | tee "$LOGDIR/run_s${SEED:-1337}.log"

echo ""
echo "============================================"
echo "  PR #254 Clone Complete."
echo "============================================"
echo "  Target: 1.1313 BPB (3-seed mean)"
f="$LOGDIR/run_s${SEED:-1337}.log"
for label in ttt_sliding sliding_window int8_zlib_roundtrip; do
    bpb=$(grep -oP "final_${label}\S* val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: $bpb" || true
done
steps=$(grep -oP 'stopping_early.*step:\K\d+' "$f" 2>/dev/null | tail -1)
size=$(grep -oP 'Total submission size\S*: \K\d+' "$f" 2>/dev/null | tail -1)
echo "  steps=${steps:-N/A} bytes=${size:-N/A}"
