#!/bin/bash
# Phase 1: Train 1 seed, then sweep n-gram params on 8xH100
# Usage: bash sweep_ngram.sh
set -e
cd "$(dirname "$0")"
mkdir -p logs

export TTT_ENABLED=0 CANON_LAST_N=0 SWA_ENABLED=0
export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
export XSA_LAST_N=11 LEAKY_RELU=1
export MAX_WALLCLOCK_SECONDS=600

# --- Step 1: Train seed 1337 (disable ngram during training eval to save time) ---
echo "=== TRAINING seed 1337 ==="
SEED=1337 NGRAM_CACHE=0 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/sweep_train.txt
echo ""

# The training saves final_model.pt — use it for eval-only sweeps
MODEL_PATH="$(pwd)/final_model.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: final_model.pt not found"
    exit 1
fi
echo "Model saved: $MODEL_PATH"

# --- Step 2: Sweep alpha (fixed order=5) ---
echo ""
echo "=== SWEEPING ALPHA (order=5) ==="
for ALPHA in 0.10 0.15 0.20 0.25 0.30; do
    echo "--- alpha=$ALPHA order=5 ---"
    EVAL_ONLY="$MODEL_PATH" ITERATIONS=0 NGRAM_CACHE=1 NGRAM_ALPHA=$ALPHA NGRAM_ORDER=5 \
        torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/sweep_a${ALPHA}_o5.txt
    grep "final_int6_sliding_window_exact" logs/sweep_a${ALPHA}_o5.txt 2>/dev/null
    echo ""
done

# --- Step 3: Sweep order (fixed alpha=0.20) ---
echo "=== SWEEPING ORDER (alpha=0.20) ==="
for ORDER in 3 4 6 7; do
    echo "--- alpha=0.20 order=$ORDER ---"
    EVAL_ONLY="$MODEL_PATH" ITERATIONS=0 NGRAM_CACHE=1 NGRAM_ALPHA=0.20 NGRAM_ORDER=$ORDER \
        torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/sweep_a0.20_o${ORDER}.txt
    grep "final_int6_sliding_window_exact" logs/sweep_a0.20_o${ORDER}.txt 2>/dev/null
    echo ""
done

# --- Summary ---
echo ""
echo "========================================="
echo "=== NGRAM SWEEP RESULTS ==="
echo "========================================="
echo ""
echo "--- Alpha sweep (order=5) ---"
for ALPHA in 0.10 0.15 0.20 0.25 0.30; do
    printf "alpha=%-4s " "$ALPHA"
    grep "final_int6_sliding_window_exact" logs/sweep_a${ALPHA}_o5.txt 2>/dev/null | awk '{print $NF}'
done
echo ""
echo "--- Order sweep (alpha=0.20) ---"
for ORDER in 3 4 5 6 7; do
    printf "order=%-2s " "$ORDER"
    if [ "$ORDER" = "5" ]; then
        grep "final_int6_sliding_window_exact" logs/sweep_a0.20_o5.txt 2>/dev/null | awk '{print $NF}'
    else
        grep "final_int6_sliding_window_exact" logs/sweep_a0.20_o${ORDER}.txt 2>/dev/null | awk '{print $NF}'
    fi
done
echo ""
echo "=== COPY ABOVE RESULTS AND PASTE BACK ==="
