#!/usr/bin/env bash
set -euo pipefail

# EXP D + SAM (clean): TTT 8ep + stride 32 + SAM sharpness-aware TTT
# No other changes — pure SAM A/B test against exp_d/run.sh

LOGDIR="logs/exp_d_sam_clean_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  EXP D + SAM clean (rho=${TTT_SAM_RHO:-0.05})"
echo "  TTT 8ep + stride 32 + SAM only"
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
EVAL_STRIDE=32 \
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=8 \
TTT_MOMENTUM=0.9 \
TTT_SAM=1 \
TTT_SAM_RHO="${TTT_SAM_RHO:-0.05}" \
NCCL_IB_DISABLE=1 \
RUN_ID="exp_d_sam_clean_s${SEED:-1337}" \
torchrun --standalone --nproc_per_node="${NPROC:-8}" \
    sota254/train_gpt.py \
    2>&1 | tee "$LOGDIR/run_s${SEED:-1337}.log"

echo ""
echo "============================================"
echo "  EXP D + SAM clean Complete."
echo "============================================"
f="$LOGDIR/run_s${SEED:-1337}.log"
for label in ttt_sliding int6_roundtrip int6_sliding_window; do
    bpb=$(grep -oP "final_${label}\S* val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: $bpb" || true
done
