#!/usr/bin/env bash
set -euo pipefail

# Ensure flash-attn is available
if ! python -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "FA3 not found, attempting install..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -3 || true
fi

# FarnsworthEngine v2 CONSERVATIVE: Only TTT v2 + XSA improvements
# Keeps original training schedule (warmdown, fixed seq len, fixed batch)
# For isolating TTT v2 gains vs full stack

LOGDIR="logs/sota_v2_tttonly_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  v2 Conservative: TTT v2 + XSA only"
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
XSA_LAST_N=3 \
D2Z_ENABLED=0 \
SEQ_CURRICULUM=0 \
BATCH_WARMUP=0 \
TTT_ENABLED=1 \
TTT_LR=0.003 \
TTT_EPOCHS=5 \
TTT_MOMENTUM=0.3 \
TTT_COSINE_DECAY=1 \
TTT_DISCRIMINATIVE_LR=1 \
TTT_WD=0.01 \
TEMP_SCALING=1 \
MOUSSE_ENABLED=0 \
NCCL_IB_DISABLE=1 \
RUN_ID="v2_tttonly_s${SEED:-1337}" \
torchrun --standalone --nproc_per_node="${NPROC:-8}" \
    sota_v2/train_gpt.py \
    2>&1 | tee "$LOGDIR/run_s${SEED:-1337}.log"

echo ""
echo "  Done. Compare against v1 baseline (1.1313 BPB)."
f="$LOGDIR/run_s${SEED:-1337}.log"
for label in ttt_sliding sliding_window int6_roundtrip; do
    bpb=$(grep -oP "final_${label}\S* val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: $bpb" || true
done
