#!/bin/bash
# Run all experiments sequentially. Run from anywhere on the pod.
# Usage: bash /workspace/my-parameter-golf/parameter-golf/run_experiments.sh
set -e

BASE=/workspace/my-parameter-golf/parameter-golf
DATA=/workspace/my-parameter-golf/data/datasets/fineweb10B_sp1024
TOK=/workspace/my-parameter-golf/data/tokenizers/fineweb_1024_bpe.model
RECORDS=$BASE/records/track_10min_16mb

run_seed() {
    local dir=$1
    local seed=$2
    local iters=$3
    local warmdown=$4
    echo ""
    echo "========================================="
    echo "Running: $(basename $dir) | SEED=$seed | ITERATIONS=$iters"
    echo "========================================="
    cd "$dir"
    SEED=$seed TTT_ENABLED=1 EVAL_STRIDE=80 \
        ITERATIONS=$iters WARMDOWN_ITERS=$warmdown \
        DATA_PATH=$DATA TOKENIZER_PATH=$TOK \
        torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee seed${seed}.log
}

# --- Experiment 1: LongContext4096_FullSOTA (~99ms/step) ---
# 6000 steps = ~594s training, int6 export, ~16MB artifact
EXP1=$RECORDS/2026-03-24_LongContext4096_FullSOTA
run_seed $EXP1 1337 6000 1440
run_seed $EXP1 42   6000 1440
run_seed $EXP1 2025 6000 1440

# --- Experiment 2: LongContext4096_Int4_16L (~99ms/step) ---
# 5500 steps = ~545s training, int4 nibble-packed export, ~14MB artifact
EXP2=$RECORDS/2026-03-24_LongContext4096_Int4_16L_FullSOTA
run_seed $EXP2 1337 5500 1320
run_seed $EXP2 42   5500 1320
run_seed $EXP2 2025 5500 1320

# --- Experiment 3: LongContext4096_Int4_BankQAT (~99ms/step) ---
# 5500 steps = ~545s training, int4 nibble-packed export, ~14MB artifact
EXP3=$RECORDS/2026-03-25_LongContext4096_Int4_BankQAT
run_seed $EXP3 1337 5500 1320
run_seed $EXP3 42   5500 1320
run_seed $EXP3 2025 5500 1320

# --- Experiment 4: LongContext4096_Int6_QAT (~99ms/step) ---
# 6000 steps = ~594s training, int6 export, ~16MB artifact
EXP4=$RECORDS/2026-03-25_LongContext4096_Int6_QAT
run_seed $EXP4 1337 6000 1440
run_seed $EXP4 42   6000 1440
run_seed $EXP4 2025 6000 1440

echo ""
echo "========================================="
echo "All experiments complete."
echo "========================================="
