#!/bin/bash
set -euo pipefail

# v7 Smooth — proper warmdown + XSA-all
# Key change: ITERATIONS=7500 matches wallclock, warmdown actually completes
# Final LR → 0 instead of ~45% peak. Should produce smoother, lower-loss weights.

cd /workspace/parameter-golf
export PYTHONPATH="/workspace/parameter-golf/flash-attention/hopper:${PYTHONPATH:-}"
mkdir -p logs

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
LOGDIR="logs/v7_smooth_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  v7 Smooth (warmdown fix + XSA-all) — seed $SEED"
echo "  Logs: $LOGDIR"
echo "============================================"

# Training: proper warmdown
ITERATIONS=7500 \
WARMDOWN_ITERS=2500 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=1792 \
INT8_SENSITIVE="" \
TTT_OPTIMIZER=sgd \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_FREEZE_BLOCKS=2 \
TTT_EMA_DECAY=0 \
TTT_MAX_TRAIN_CHUNKS=50 \
TTT_WARMUP_CHUNKS=0 \
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
