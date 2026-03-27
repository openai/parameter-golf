#!/bin/bash
# Run all experiments sequentially. Run from anywhere on the pod.
# Usage: bash /workspace/my-parameter-golf/parameter-golf/run_experiments.sh
set -e

BASE=/workspace/my-parameter-golf

# Git identity (needed for commits on fresh pods)
git config --global user.email "runpod@parameter-golf" 2>/dev/null || true
git config --global user.name "FlashyFlash3011" 2>/dev/null || true
DATA=$BASE/data/datasets/fineweb10B_sp1024
TOK=$BASE/data/tokenizers/fineweb_1024_bpe.model
RECORDS=$BASE/records/track_10min_16mb

save_and_push() {
    local dir=$1
    local seed=$2
    local exp_name=$(basename $dir)
    cd $BASE
    git add "$dir/seed${seed}.log" 2>/dev/null || true
    if ! git diff --cached --quiet; then
        git commit -m "results: ${exp_name} seed${seed}"
        git push fork flashyflash3011/long-context-4096-qat-int4-16l || \
            echo "WARNING: push failed, log is committed locally"
    fi
}

run_seed() {
    local dir=$1
    local seed=$2
    local iters=$3
    local warmdown=$4
    local stride=${5:-80}
    local extra_env=${6:-""}
    echo ""
    echo "========================================="
    echo "Running: $(basename $dir) | SEED=$seed | ITERATIONS=$iters | EVAL_STRIDE=$stride"
    echo "========================================="
    cd "$dir"
    env SEED=$seed TTT_ENABLED=1 EVAL_STRIDE=$stride \
        ITERATIONS=$iters WARMDOWN_ITERS=$warmdown \
        DATA_PATH=$DATA TOKENIZER_PATH=$TOK \
        $extra_env \
        torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee seed${seed}.log
    save_and_push "$dir" "$seed"
}

# --- Experiment 1: LongContext4096_FullSOTA (~99ms/step) ---
# 6000 steps = ~594s training, int6 export, ~16MB artifact
EXP1=$RECORDS/2026-03-24_LongContext4096_FullSOTA
run_seed $EXP1 1337 6000 1440
run_seed $EXP1 42   6000 1440
run_seed $EXP1 2025 6000 1440

# --- Experiment 2: LongContext4096_Int4_16L (~140ms/step) ---
# 5500 steps = ~770s training (wallclock cap ~4277 steps), int4 export, ~14MB artifact
# EVAL_STRIDE=96 keeps TTT eval ~430s (was 697s at stride=80, over 600s limit)
EXP2=$RECORDS/2026-03-24_LongContext4096_Int4_16L_FullSOTA
run_seed $EXP2 1337 5500 1320 96
run_seed $EXP2 42   5500 1320 96
run_seed $EXP2 2025 5500 1320 96

# --- Experiment 3: QAT_Int4_16L_FullSOTA (~140ms/step) ---
# Same as Exp 2 but with QAT_ENABLED=1 (int4 fake-quant from step 1, not just late warmdown)
# EVAL_STRIDE=96 for same TTT timing fix
EXP3=$RECORDS/2026-03-24_QAT_Int4_16L_FullSOTA
run_seed $EXP3 1337 5500 1320 96 "QAT_ENABLED=1"
run_seed $EXP3 42   5500 1320 96 "QAT_ENABLED=1"
run_seed $EXP3 2025 5500 1320 96 "QAT_ENABLED=1"

# --- Experiment 4: LongContext4096_Int4_BankQAT (~140ms/step) ---
# 5500 steps, int4+bank QAT (risky), EVAL_STRIDE=96 for TTT timing fix
EXP4=$RECORDS/2026-03-25_LongContext4096_Int4_BankQAT
run_seed $EXP4 1337 5500 1320 96
run_seed $EXP4 42   5500 1320 96
run_seed $EXP4 2025 5500 1320 96

# --- Experiment 5: LongContext4096_Int6_QAT (~99ms/step) ---
# 6000 steps = ~594s training, int6 QAT from step 1, ~16MB artifact (QAT_ENABLED=1 is default in script)
# TTT eval ~515s at stride=80 (11L, within 600s limit)
EXP5=$RECORDS/2026-03-25_LongContext4096_Int6_QAT
run_seed $EXP5 1337 6000 1440
run_seed $EXP5 42   6000 1440
run_seed $EXP5 2025 6000 1440

echo ""
echo "========================================="
echo "All experiments complete."
echo "========================================="
