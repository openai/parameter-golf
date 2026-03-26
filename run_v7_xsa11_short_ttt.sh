#!/bin/bash
set -euo pipefail

# v7 XSA-all + Short TTT — targeting 1st place
# Changes from PR #508:
#   - XSA on all 11 layers (was 4) — PR #503 proves this helps
#   - Short TTT: SGD, no EMA, 50 chunks (capture chunk-51 peak)
#   - Training EMA already 0.997 (matches #505/#503)
# Base: same v7 relu² arch, GPTQ, early QAT

cd /workspace/parameter-golf
export PYTHONPATH="/workspace/parameter-golf/flash-attention/hopper:${PYTHONPATH:-}"
mkdir -p logs

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
LOGDIR="logs/v7_xsa11_short_ttt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  v7 XSA-all + Short TTT — seed $SEED"
echo "  Logs: $LOGDIR"
echo "============================================"

# Architecture change: XSA on all 11 layers
XSA_LAST_N=11 \
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
