#!/usr/bin/env bash
set -euo pipefail

# EXP B: SwiGLU MLP replacing ReLU²
# gate(x) * up(x) with SiLU activation → consistently better in LLaMA/Mistral.
# hidden=1024 (2/3 * 1536) matches ReLU² param count exactly.

LOGDIR="logs/exp_b_swiglu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  EXP B: SwiGLU MLP on SOTA 254 base"
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
RUN_ID="exp_b_swiglu_s${SEED:-1337}" \
torchrun --standalone --nproc_per_node="${NPROC:-8}" \
    exp_b/train_gpt.py \
    2>&1 | tee "$LOGDIR/run_s${SEED:-1337}.log"

echo ""
echo "============================================"
echo "  EXP B Complete."
echo "============================================"
f="$LOGDIR/run_s${SEED:-1337}.log"
for label in int6_roundtrip int6_sliding_window; do
    bpb=$(grep -oP "final_${label}\S* val_loss:\S+ val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: $bpb" || true
done
