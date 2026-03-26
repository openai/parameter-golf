#!/bin/bash
set -euo pipefail

# v7 Short TTT Experiment — Option A (no EMA, 50 chunks, SGD)
# Tests capturing the chunk-51 peak without EMA dilution
# Base model is identical to PR #508 (1.1206 BPB, 15.56MB)

cd /workspace/parameter-golf
export PYTHONPATH="/workspace/parameter-golf/flash-attention/hopper:${PYTHONPATH:-}"
mkdir -p logs

# Verify deps
python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
LOGDIR="logs/v7_short_ttt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  v7 Short TTT — seed $SEED"
echo "  Logs: $LOGDIR"
echo "============================================"

# Training: identical to PR #508
# TTT: SGD, short window, no EMA
TTT_OPTIMIZER=sgd \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_FREEZE_BLOCKS=2 \
TTT_EMA_DECAY=0 \
TTT_MAX_TRAIN_CHUNKS=50 \
TTT_WARMUP_CHUNKS=0 \
INT8_SENSITIVE="" \
SEED="$SEED" \
torchrun --standalone --nproc_per_node=8 \
    train_gpt_v7.py \
    2>&1 | tee "$LOGDIR/run_s${SEED}.log"

echo ""
echo "============================================"
echo "  Done — seed $SEED"
echo "============================================"
f="$LOGDIR/run_s${SEED}.log"
for label in final_int6_sliding_window_exact legal_ttt_exact; do
    bpb=$(grep -oP "${label} val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: $bpb" || true
done
