#!/bin/bash
set -e

MODE=${1:-1h100}

# Validate and parse mode
if [[ "$MODE" =~ ^([0-9]+)([ah]100)$ ]]; then
    NUM_GPUS=${BASH_REMATCH[1]}
    ARCH=${BASH_REMATCH[2]}
else
    echo "Invalid mode: $MODE. Format should be <N>h100 or <N>a100 (e.g., 1h100, 4h100, 8a100)."
    exit 1
fi

echo "================================================================="
echo "PRISM: WT Improvements Ablation (2500 STEPS)"
echo "Mode: $MODE ($NUM_GPUS GPUs, Architecture: $ARCH)"
echo "================================================================="

echo "================================================================="
echo "Installing Dependencies"
echo "================================================================="
pip install -q brotli sentencepiece numpy tqdm huggingface-hub datasets tiktoken typing-extensions==4.15.0 setuptools kernels

if [[ "$ARCH" != "a100" ]]; then
    pip install flash_attn_3 --no-deps \
      --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ \
      2>/dev/null || echo "WARN: flash_attn_3 wheel install failed"
else
    echo "A100 Mode selected: Skipping flash_attn_3 (Hopper-only) installation. Will use PyTorch SDPA fallback."
fi

echo "================================================================="
echo "Downloading SP8192 tokenizer & FineWeb shards"
echo "================================================================="
# Navigate to parameter-golf root to run the data download script
cd /workspace/parameter-golf
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192
# Return to the workshop folder (using cd - to return to previous dir)
cd -

mkdir -p logs_exp_wt
mkdir -p models_exp_wt

export MAX_WALLCLOCK_SECONDS=0
export OMP_NUM_THREADS=4

run_job() {
    local gpu=$1
    local run_id=$2
    local adapt=$3
    local anchor=$4
    local enc_only=$5
    local perm_heads=$6
    
    echo "================================================================="
    echo "Starting: $run_id on GPU $gpu"
    echo "================================================================="
    
    local use_a100=0
    if [[ "$ARCH" == "a100" ]]; then
        use_a100=1
    fi
    
    CUDA_VISIBLE_DEVICES=$gpu \
    USE_A100=$use_a100 \
    RUN_ID=$run_id \
    DATA_DIR="/workspace/parameter-golf/data" \
    NUM_LOOPS=0 \
    PARALLEL_RESIDUAL=0 \
    VAL_LOSS_EVERY=500 \
    TRAIN_LOG_EVERY=100 \
    ITERATIONS=2500 \
    SLIDING_WINDOW_ENABLED=0 \
    SHARED_ENCDEC_ENABLED=1 \
    ALBERT_MODE=0 \
    NAIVE_SHARED_MODE=0 \
    REPARTITION_ENABLED=1 \
    ADAPTIVE_REPARTITION_ENABLED=$adapt \
    ANCHOR_HEADS=$anchor \
    PERM_ENCODER_ONLY=$enc_only \
    REPARTITION_PERMUTE_HEADS=$perm_heads \
    SEED=42 \
    LOGFILE="logs_exp_wt/${run_id}.txt" \
    MODEL_PATH="models_exp_wt/final_model_${run_id}.pt" \
    QUANTIZED_MODEL_PATH="models_exp_wt/final_model_${run_id}.int6.ptz" \
    python3 train_gpt_exp_wt.py > "logs_exp_wt/${run_id}_console.log" 2>&1
    
    echo "Finished: $run_id on GPU $gpu"
}

# Keep track of active jobs and GPU assignment
active_jobs=0
gpu_idx=0

launch_experiment() {
    local run_id=$1
    local adapt=$2
    local anchor=$3
    local enc_only=$4
    local perm_heads=$5
    
    # Run the job in the background on the next available GPU
    run_job $gpu_idx $run_id $adapt $anchor $enc_only $perm_heads &
    
    active_jobs=$((active_jobs + 1))
    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
    
    # If we've reached the GPU limit, wait for all background jobs to finish
    if [[ $active_jobs -eq $NUM_GPUS ]]; then
        wait
        active_jobs=0
    fi
}

echo "Launching experiments across $NUM_GPUS GPUs..."

# Proposal 1: Partial Head Permutation (Anchor 4 heads, permute remaining 4)
launch_experiment "prism_wt_partial4_seed42" 0 4 0 1
launch_experiment "prism_adapt_partial4_seed42" 1 4 0 1

# Proposal 2: Encoder-Only Permutation (Identity for decoder layers)
launch_experiment "prism_wt_enc_only_seed42" 0 0 1 1
launch_experiment "prism_adapt_enc_only_seed42" 1 0 1 1

# Proposal 3: Dims-Only Permutation (No head reordering, just within-head shuffle)
launch_experiment "prism_wt_dims_only_seed42" 0 0 0 0
launch_experiment "prism_adapt_dims_only_seed42" 1 0 0 0

# Wait for any remaining background jobs to finish
wait

echo "All 6 experiments completed successfully."
