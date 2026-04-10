#!/bin/bash
# ============================================================
# Raki A/B/C Test — V5 vs V7 vs V8 (RunPod 1xGPU, 5min each)
# ============================================================
set -e

SECS=300
COMMON="MUON_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=500 \
EMA_DECAY=0.997 EVAL_STRIDE=64 TRAIN_BATCH_TOKENS=786432 \
MAX_WALLCLOCK_SECONDS=$SECS WARMUP_STEPS=10 VAL_LOSS_EVERY=500 SEED=1337"

echo "============================================"
echo "  Raki Comparison — 5min × 3 runs (1xGPU)"
echo "============================================"

# --- data ---
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    echo "[DATA] Downloading..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
else
    echo "[DATA] Already present."
fi

mkdir -p logs
cp train_gpt.py train_gpt_backup.py

run_version() {
    local VER=$1
    local PATCH=$2
    local EXTRA=$3
    echo ""
    echo "===== Running $VER ====="
    cp train_gpt_backup.py train_gpt.py
    python3 $PATCH
    env $COMMON $EXTRA RUN_ID=test_$VER \
        torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/test_$VER.txt
}

# --- V5 ---
run_version "v5" "patch_v5.py" ""

# --- V7 ---
run_version "v7" "patch_v7.py" "MLP_MULT=3"

# --- V8 ---
run_version "v8" "patch_v8.py" ""

# --- restore ---
cp train_gpt_backup.py train_gpt.py

# --- results ---
echo ""
echo "============================================"
echo "  RESULTS"
echo "============================================"
printf "%-6s  %-14s  %-14s  %-10s  %-8s\n" "Ver" "Pre-quant BPB" "Post-quant BPB" "Quant gap" "Steps"

for V in v5 v7 v8; do
    LOG="logs/test_$V.txt"
    PRE=$(grep "val_bpb" $LOG | grep -v "final" | tail -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/' 2>/dev/null || echo "?")
    POST=$(grep "roundtrip_exact" $LOG | sed 's/.*val_bpb:\([0-9.]*\).*/\1/' 2>/dev/null || echo "?")
    STEP=$(grep "stopping_early\|^step:" $LOG | tail -1 | sed 's/.*step:\([0-9]*\).*/\1/' 2>/dev/null || echo "?")
    if [[ "$PRE" != "?" && "$POST" != "?" ]]; then
        GAP=$(python3 -c "print(f'{float(\"$POST\")-float(\"$PRE\"):.4f}')" 2>/dev/null || echo "?")
    else
        GAP="?"
    fi
    printf "%-6s  %-14s  %-14s  %-10s  %-8s\n" "$V" "$PRE" "$POST" "$GAP" "$STEP"
done

echo ""
echo "V8 quant gap << V5 quant gap = Late QAT calisiyor"
echo "V8 pre-quant < V5 pre-quant = LeakyReLU²+XSA+LN etkisi"
echo "============================================"
