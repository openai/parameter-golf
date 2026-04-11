#!/usr/bin/env bash
set -euo pipefail

# FarnsworthEngine v2: Full improvement stack on top of PR #254 SOTA (1.1313 BPB)
#
# Changes from v1:
#   Training: D2Z LR schedule, seq-length curriculum (256→2048), batch warmup (262K→786K)
#   Eval:     TTT v2 (cosine decay + discriminative LR + low momentum), temperature scaling
#   Arch:     XSA last 3 layers
#   Optional: Mousse optimizer (MOUSSE_ENABLED=1)
#
# Target: < 1.120 BPB

LOGDIR="logs/sota_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  FarnsworthEngine v2 — Full Stack"
echo "  Base: PR #254 (1.1313 BPB)"
echo "  + TTT v2 + Curriculum + D2Z + XSA + TempScale"
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
D2Z_ENABLED=1 \
D2Z_WARMUP_STEPS=200 \
SEQ_CURRICULUM=1 \
SEQ_CURRICULUM_MIN=256 \
SEQ_CURRICULUM_RAMP_FRAC=0.25 \
BATCH_WARMUP=1 \
BATCH_WARMUP_START=262144 \
BATCH_WARMUP_STEPS=1000 \
TTT_ENABLED=1 \
TTT_LR=0.003 \
TTT_EPOCHS=5 \
TTT_MOMENTUM=0.3 \
TTT_COSINE_DECAY=1 \
TTT_DISCRIMINATIVE_LR=1 \
TTT_WD=0.01 \
TEMP_SCALING=1 \
MOUSSE_ENABLED="${MOUSSE_ENABLED:-0}" \
NCCL_IB_DISABLE=1 \
RUN_ID="v2_s${SEED:-1337}" \
torchrun --standalone --nproc_per_node="${NPROC:-8}" \
    sota_v2/train_gpt.py \
    2>&1 | tee "$LOGDIR/run_s${SEED:-1337}.log"

echo ""
echo "============================================"
echo "  FarnsworthEngine v2 Complete."
echo "============================================"
echo "  Baseline: 1.1313 BPB (v1, PR #254)"
f="$LOGDIR/run_s${SEED:-1337}.log"
for label in ttt_sliding sliding_window int6_roundtrip; do
    bpb=$(grep -oP "final_${label}\S* val_bpb:\K\S+" "$f" 2>/dev/null | tail -1)
    [ -n "$bpb" ] && echo "  ${label}: $bpb" || true
done
temp=$(grep -oP "temp_scaling:done T=\K\S+" "$f" 2>/dev/null | tail -1)
[ -n "$temp" ] && echo "  temperature: $temp" || true
steps=$(grep -oP 'stopping_early.*step:\K\d+' "$f" 2>/dev/null | tail -1)
size=$(grep -oP 'Total submission size\S*: \K\d+' "$f" 2>/dev/null | tail -1)
echo "  steps=${steps:-N/A} bytes=${size:-N/A}"
