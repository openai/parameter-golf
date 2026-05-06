#!/bin/bash
set -e

# ==============================================================================
# PRISM: NeurIPS Workshop Experiments - TOKEN LIMITED (2000 STEPS)
# ==============================================================================
# This script runs all experiments capped precisely at 2000 steps.
# It uses the V3 Adapt logic. Sliding window evaluation is disabled.
# ==============================================================================

echo "================================================================="
echo "Installing Dependencies"
echo "================================================================="
pip install -q brotli sentencepiece numpy tqdm huggingface-hub datasets tiktoken typing-extensions==4.15.0 setuptools kernels
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ \
  2>/dev/null || echo "WARN: flash_attn_3 wheel install failed"

echo "================================================================="
echo "Downloading SP8192 tokenizer & FineWeb shards"
echo "================================================================="
# Navigate to parameter-golf root to run the data download script
cd /workspace/parameter-golf
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192
# Return to the workshop folder
cd /workspace/parameter-golf/records/track_10min_16mb/neurips_workshop || cd -

# Create organized output directories specifically for token-limited runs
mkdir -p logs_token_limited
mkdir -p models_token_limited

# Define the common constraints
# Remove wallclock constraint to guarantee exactly 2000 steps are hit based on tokens, not time.
# Setting to 0 correctly disables the time-based LR scheduler and falls back to step-based scheduling.
export MAX_WALLCLOCK_SECONDS=0
export OMP_NUM_THREADS=4

# Base command wrapper function
# Arguments: GPU_ID RUN_ID SHARED ALBERT NAIVE REPART ADAPT SEED
run_job() {
    local gpu=$1
    local run_id=$2
    local shared=$3
    local albert=$4
    local naive=$5
    local repart=$6
    local adapt=$7
    local seed=$8
    
    echo "[GPU $gpu] Starting: $run_id"
    
    # We pass MODEL_PATH explicitly so parallel jobs don't overwrite each other's weights
    CUDA_VISIBLE_DEVICES=$gpu \
    RUN_ID=$run_id \
    DATA_DIR="/workspace/parameter-golf/data" \
    NUM_LOOPS=0 \
    PARALLEL_RESIDUAL=0 \
    VAL_LOSS_EVERY=500 \
    TRAIN_LOG_EVERY=100 \
    ITERATIONS=2500 \
    SLIDING_WINDOW_ENABLED=0 \
    SHARED_ENCDEC_ENABLED=$shared \
    ALBERT_MODE=$albert \
    NAIVE_SHARED_MODE=$naive \
    REPARTITION_ENABLED=$repart \
    ADAPTIVE_REPARTITION_ENABLED=$adapt \
    SEED=$seed \
    LOGFILE="logs_token_limited/${run_id}.txt" \
    MODEL_PATH="models_token_limited/final_model_${run_id}.pt" \
    QUANTIZED_MODEL_PATH="models_token_limited/final_model_${run_id}.int6.ptz" \
    python3 train_gpt_prism_token_limit.py > "logs_token_limited/${run_id}_console.log" 2>&1
    
    echo "[GPU $gpu] Finished: $run_id"
}

export -f run_job

echo "================================================================="
echo "Starting Batch 1/3 (8 Parallel Jobs)..."
echo "================================================================="
run_job 0 "standard11_seed42" 0 0 0 0 0 42 &
run_job 1 "standard11_seed1337" 0 0 0 0 0 1337 &
run_job 2 "standard11_seed2024" 0 0 0 0 0 2024 &
run_job 3 "albert_seed42" 1 1 0 0 0 42 &
run_job 4 "albert_seed1337" 1 1 0 0 0 1337 &
run_job 5 "albert_seed2024" 1 1 0 0 0 2024 &
run_job 6 "naive_shared_seed42" 1 0 1 0 0 42 &
run_job 7 "naive_shared_seed1337" 1 0 1 0 0 1337 &

# Wait for all jobs in Batch 1 to finish
wait
echo "Batch 1 Completed."

echo "================================================================="
echo "Starting Batch 2/3 (8 Parallel Jobs)..."
echo "================================================================="
run_job 0 "naive_shared_seed2024" 1 0 1 0 0 2024 &
run_job 1 "prism_wo_seed42" 1 0 0 0 0 42 &
run_job 2 "prism_wo_seed1337" 1 0 0 0 0 1337 &
run_job 3 "prism_wo_seed2024" 1 0 0 0 0 2024 &
run_job 4 "prism_wt_seed42" 1 0 0 1 0 42 &
run_job 5 "prism_wt_seed1337" 1 0 0 1 0 1337 &
run_job 6 "prism_wt_seed2024" 1 0 0 1 0 2024 &
run_job 7 "prism_adapt_v3_seed42" 1 0 0 1 1 42 &

# Wait for all jobs in Batch 2 to finish
wait
echo "Batch 2 Completed."

echo "================================================================="
echo "Starting Batch 3/3 (2 Parallel Jobs)..."
echo "================================================================="
run_job 0 "prism_adapt_v3_seed1337" 1 0 0 1 1 1337 &
run_job 1 "prism_adapt_v3_seed2024" 1 0 0 1 1 2024 &

# Wait for all jobs in Batch 3 to finish
wait
echo "Batch 3 Completed."

echo "================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "Outputs are securely stored in logs_token_limited/ and models_token_limited/"
echo "================================================================="
