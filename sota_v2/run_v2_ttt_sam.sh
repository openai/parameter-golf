#!/usr/bin/env bash
set -euo pipefail

# TTT with SAM (Sharpness-Aware Minimization)
# Tests if TTT failure is a sharpness/generalization problem

LOGDIR="logs/sota_v2_ttt_sam_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  v2: TTT SAM (rho=${TTT_SAM_RHO:-0.05})"
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
XSA_LAST_N=0 \
D2Z_ENABLED=0 \
SEQ_CURRICULUM=0 \
BATCH_WARMUP=0 \
TTT_ENABLED=1 \
TTT_LR="${TTT_LR:-0.002}" \
TTT_EPOCHS="${TTT_EPOCHS:-3}" \
TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}" \
TTT_COSINE_DECAY=0 \
TTT_DISCRIMINATIVE_LR=0 \
TTT_WD=0 \
TTT_SAM=1 \
TTT_SAM_RHO="${TTT_SAM_RHO:-0.05}" \
TEMP_SCALING=0 \
MOUSSE_ENABLED=0 \
NCCL_IB_DISABLE=1 \
RUN_ID="v2_ttt_sam_s${SEED:-1337}" \
torchrun --standalone --nproc_per_node="${NPROC:-8}" \
    sota_v2/train_gpt.py \
    2>&1 | tee "$LOGDIR/run_s${SEED:-1337}.log"

echo ""
echo "  Done. Compare against v1 baseline (1.1301 BPB)."
f="$LOGDIR/run_s${SEED:-1337}.log"
for label in ttt_sliding sliding_window int6_roundtrip; do
    bpb=$(grep -oP "final_${label}\S* val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: $bpb" || true
done
