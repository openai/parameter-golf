#!/bin/bash
#SBATCH --job-name=pr940_nflow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --nice=0
#SBATCH --output=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/logs/pr940_nflow_%j.out
#SBATCH --error=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/logs/pr940_nflow_%j.err
#SBATCH --chdir=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf
#SBATCH --account=medcam

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "================"

# Activate environment
source /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/.venv/bin/activate

# Force single GPU
export CUDA_VISIBLE_DEVICES=0

# --- Run & Data ---
export RUN_ID="pr940_nflow_${SLURM_JOB_ID}"
export SEED=42
export DATA_PATH="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/repo/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/repo/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024

# --- Training Schedule ---
export MAX_WALLCLOCK_SECONDS=0
export ITERATIONS=7000
export VAL_LOSS_EVERY=500
export WARMDOWN_ITERS=2800
export WARMUP_STEPS=20
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048

# --- Architecture ---
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export TIE_EMBEDDINGS=1
export BIGRAM_VOCAB_SIZE=4096
export BIGRAM_DIM=128
export ROPE_DIMS=16
export ROPE_BASE=10000
export LOGIT_SOFTCAP=30.0
export LN_SCALE=1
export XSA_LAST_N=11
export VALUE_RESIDUAL=1
export GATED_ATTENTION=1
export QK_GAIN_INIT=1.5
export LEAKY_RELU=1
export LEAKY_SLOPE=0.5

# --- Optimizer ---
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3
export EVAL_STRIDE=64

# --- EMA ---
export EMA_ENABLED=1
export EMA_DECAY=0.997

# --- Disabled features ---
export TTT_ENABLED=0
export CANON_LAST_N=0
export SWA_ENABLED=0
export QAT_ENABLED=0

# --- FlowRefiner (disabled) ---
export FLOW_ENABLED=0

# --- NativeFlowMatcher ---
export NATIVE_FLOW_ENABLED=1
export NATIVE_FLOW_HIDDEN_DIM=256
export NATIVE_FLOW_INIT_SCALE=0.01
export NATIVE_FLOW_LOSS_WEIGHT=0.1

# --- Log file ---
LOGFILE="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/logs/pr940_nflow_${SLURM_JOB_ID}.txt"

echo "Running training with RUN_ID=$RUN_ID"
echo "Log: $LOGFILE"

torchrun --standalone --nproc_per_node=1 \
    train_gpt_pr940.py 2>&1 | tee "$LOGFILE"

echo "=== Done: $(date) ==="
