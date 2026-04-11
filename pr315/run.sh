#!/usr/bin/env bash
set -euo pipefail

# PR#315 + TTT 8ep SAM: Partial RoPE + LN Scale + EMA + XSA4 + TTT + SAM
# Target: beat 1.1248 BPB (PR#315 baseline without TTT)

LOGDIR="logs/pr315_ttt8_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  PR#315 + TTT 8ep (seed ${SEED:-1337})"
echo "  Logs: $LOGDIR"
echo "============================================"

SEED="${SEED:-1337}" \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
SWA_ENABLED=0 \
ROPE_DIMS=16 \
LN_SCALE=1 \
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
TTT_EPOCHS=8 \
TTT_MOMENTUM=0.9 \
TTT_SAM=1 \
TTT_SAM_RHO=0.05 \
NCCL_IB_DISABLE=1 \
RUN_ID="pr315_ttt8_s${SEED:-1337}" \
torchrun --standalone --nproc_per_node="${NPROC:-8}" \
    pr315/train_gpt.py \
    2>&1 | tee "$LOGDIR/run_s${SEED:-1337}.log"

echo ""
echo "============================================"
echo "  PR#315 + TTT Complete."
echo "============================================"
f="$LOGDIR/run_s${SEED:-1337}.log"
for label in ttt_sliding int6_roundtrip int6_sliding_window; do
    bpb=$(grep -oP "final_${label}\S* val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: $bpb" || true
done
