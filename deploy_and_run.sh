#!/usr/bin/env bash
# =================================================================
# ONE-COMMAND DEPLOY: Paste this in RunPod web terminal and walk away
# Does everything: setup → SOTA reproduction → SLOT improvement → 3-seed validation
# =================================================================
set -euo pipefail

FORK="https://github.com/Omrigotlieb/parameter-golf.git"
RESULTS="/workspace/results"
mkdir -p "$RESULTS"

echo "=== PARAMETER GOLF DEPLOY $(date) ===" | tee "$RESULTS/status.txt"

# --- STEP 1: Setup ---
cd /workspace/parameter-golf
echo "Step 1: Fetching our improvements..." | tee -a "$RESULTS/status.txt"
git remote add fork "$FORK" 2>/dev/null || true
git fetch fork main --quiet
git checkout fork/main -- records/track_10min_16mb/2026-04-02_PolarExpress_SLOT_XSAall/

# Check data
SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
if [ "$SHARDS" -lt 10 ]; then
    echo "Downloading data..." | tee -a "$RESULTS/status.txt"
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
fi

# Check GPUs
NGPU=$(nvidia-smi -L | wc -l)
echo "GPUs: $NGPU" | tee -a "$RESULTS/status.txt"

SCRIPT="records/track_10min_16mb/2026-04-02_PolarExpress_SLOT_XSAall/train_gpt.py"

# --- STEP 2: Smoke test (200 iters, ~2 min) ---
echo "" | tee -a "$RESULTS/status.txt"
echo "=== SMOKE TEST ===" | tee -a "$RESULTS/status.txt"
ITERATIONS=200 MAX_WALLCLOCK_SECONDS=0 SEED=1337 \
torchrun --standalone --nproc_per_node=$NGPU "$SCRIPT" 2>&1 | tee "$RESULTS/smoke.log" | tail -5
echo "Smoke test done $(date)" | tee -a "$RESULTS/status.txt"

# --- STEP 3: Full SOTA reproduction (10 min) ---
echo "" | tee -a "$RESULTS/status.txt"
echo "=== RUN 1: SOTA REPRODUCTION (seed=1337) ===" | tee -a "$RESULTS/status.txt"
echo "Started: $(date)" | tee -a "$RESULTS/status.txt"

BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 \
MUON_BACKEND_STEPS=5 \
SLOT_ENABLED=0 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=$NGPU "$SCRIPT" 2>&1 | tee "$RESULTS/run1_sota.log"

R1=$(grep "final_int6_sliding_window_s64 " "$RESULTS/run1_sota.log" | tail -1 | sed 's/.*val_bpb://' | awk '{print $1}')
echo "RUN 1 BPB: $R1" | tee -a "$RESULTS/status.txt"

# --- STEP 4: Our improvements (Polar Express + SLOT) ---
echo "" | tee -a "$RESULTS/status.txt"
echo "=== RUN 2: POLAR EXPRESS + SLOT (seed=1337) ===" | tee -a "$RESULTS/status.txt"
echo "Started: $(date)" | tee -a "$RESULTS/status.txt"

BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 \
MUON_BACKEND_STEPS=4 \
SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.005 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=$NGPU "$SCRIPT" 2>&1 | tee "$RESULTS/run2_slot.log"

R2_PRE=$(grep "final_int6_sliding_window_s64 " "$RESULTS/run2_slot.log" | tail -1 | sed 's/.*val_bpb://' | awk '{print $1}')
R2_SLOT=$(grep "slot_eval " "$RESULTS/run2_slot.log" | tail -1 | sed 's/.*val_bpb://' | awk '{print $1}')
echo "RUN 2 BPB: pre_slot=$R2_PRE post_slot=$R2_SLOT" | tee -a "$RESULTS/status.txt"

# --- STEP 5: Additional seeds ---
echo "" | tee -a "$RESULTS/status.txt"
echo "=== 3-SEED VALIDATION ===" | tee -a "$RESULTS/status.txt"

for SEED in 42 2025; do
    echo "Seed $SEED started: $(date)" | tee -a "$RESULTS/status.txt"
    BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
    XSA_LAST_N=11 WARMDOWN_ITERS=4000 \
    MUON_BACKEND_STEPS=4 \
    SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.005 \
    ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
    EVAL_STRIDE=64 \
    SEED=$SEED \
    torchrun --standalone --nproc_per_node=$NGPU "$SCRIPT" 2>&1 | tee "$RESULTS/run3_seed${SEED}.log"

    S_PRE=$(grep "final_int6_sliding_window_s64 " "$RESULTS/run3_seed${SEED}.log" | tail -1 | sed 's/.*val_bpb://' | awk '{print $1}')
    S_SLOT=$(grep "slot_eval " "$RESULTS/run3_seed${SEED}.log" | tail -1 | sed 's/.*val_bpb://' | awk '{print $1}')
    echo "Seed $SEED BPB: pre_slot=$S_PRE post_slot=$S_SLOT" | tee -a "$RESULTS/status.txt"
done

echo "" | tee -a "$RESULTS/status.txt"
echo "=== ALL DONE $(date) ===" | tee -a "$RESULTS/status.txt"
echo ""
echo "========================================"
echo "  FINAL RESULTS"
echo "========================================"
cat "$RESULTS/status.txt"
