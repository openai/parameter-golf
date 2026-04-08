#!/bin/bash
# run_all_benchmarks.sh
# Comprehensive ablation and stability suite. Runs sequentially to avoid OOM.
# Outputs to benchmark_results/

mkdir -p benchmark_results

# Command parameters for a fast 4L/256d toy config that matches the PR runs
BASE_CONF="DATA_PATH=/tmp/pg_data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/tmp/pg_data/tokenizers/fineweb_1024_bpe.model NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 ITERATIONS=100000"

export PYTHON_BIN="/opt/homebrew/bin/python3"

echo "Starting ablation study (20 min budget per run)..."

run_experiment() {
    local name=$1
    local config=$2
    local seed=$3
    local time_limit=$4
    local out_file="benchmark_results/${name}_seed${seed}.log"
    echo "Running $name (Seed $seed, Limit ${time_limit}s)..."
    
    # Run in foreground but under caffeinate to prevent sleep
    caffeinate -i bash -c "cd /Users/akhileshgogikar/parameter-golf/records/track_10min_16mb/2026-03-25_Ternary_Feedback_TTT && env $BASE_CONF $config SEED=$seed MAX_WALLCLOCK_SECONDS=$time_limit bash run_mlx_reasoner.sh" > "$out_file" 2>&1
    
    # Parse final BPB
    local bpb=$(grep -E "^final_sliding|final_eval" "$out_file" | tail -n 1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/')
    echo "-> Finished $name: BPB = $bpb"
    echo "$name,$seed,$bpb" >> benchmark_results/summary.csv
}

# 1. Ablation runs (1200 seconds / 20 mins)
# Baseline: Plain Ternary
run_experiment "A_PlainTernary" "FEEDBACK_ENABLED=0 BIGRAM_HASH_ENABLED=0 VRL_ENABLED=0 XSA_START_LAYER=4 CAPSULE_ENABLED=0 KOOPMAN_ENABLED=0 TURBO_QUANT_KV=0" 42 1200

# B: Base + Feedback + Engram + VRL + XSA
run_experiment "B_CoreTricks" "FEEDBACK_ENABLED=1 BIGRAM_HASH_ENABLED=1 VRL_ENABLED=1 XSA_START_LAYER=2 CAPSULE_ENABLED=0 KOOPMAN_ENABLED=0 TURBO_QUANT_KV=0" 42 1200

# C: Base + Tricks + Capsules (No Koopman)
run_experiment "C_CapsulesNoKoopman" "FEEDBACK_ENABLED=1 BIGRAM_HASH_ENABLED=1 VRL_ENABLED=1 XSA_START_LAYER=2 CAPSULE_ENABLED=1 KOOPMAN_ENABLED=0 TURBO_QUANT_KV=0" 42 1200

# D: Base + Tricks + KoopCaps
run_experiment "D_KoopCaps" "FEEDBACK_ENABLED=1 BIGRAM_HASH_ENABLED=1 VRL_ENABLED=1 XSA_START_LAYER=2 CAPSULE_ENABLED=1 KOOPMAN_ENABLED=1 TURBO_QUANT_KV=0" 42 1200

# E: Full Architecture (KoopCaps + TurboQuant)
run_experiment "E_FullStack" "FEEDBACK_ENABLED=1 BIGRAM_HASH_ENABLED=1 VRL_ENABLED=1 XSA_START_LAYER=2 CAPSULE_ENABLED=1 KOOPMAN_ENABLED=1 TURBO_QUANT_KV=1" 42 1200


echo "Starting 5-seed stability run on E_FullStack (30 min budget per run)..."

# 2. Stability runs (1800 seconds / 30 mins)
# Seed 42 is already done essentially, but we re-run to match the 30-min horizon.
for seed in 42 1337 7 2024 999; do
    run_experiment "Stability_FullStack" "FEEDBACK_ENABLED=1 BIGRAM_HASH_ENABLED=1 VRL_ENABLED=1 XSA_START_LAYER=2 CAPSULE_ENABLED=1 KOOPMAN_ENABLED=1 TURBO_QUANT_KV=1" $seed 1800
done

echo "Running report generator..."
$PYTHON_BIN generate_report.py

echo "All benchmarks completed!"
