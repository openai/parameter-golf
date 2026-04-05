#!/bin/bash
#SBATCH --job-name=gdn_7k_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/logs/7k_v2_%j.out
#SBATCH --error=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/logs/7k_v2_%j.err
#SBATCH --chdir=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments

# ============================================================================
# Model A PureGDN — 7000 step FRESH training run (v2, 10 layers)
# 2×A100 FALLBACK, longer wallclock
# QAT bug fixed, clean start from scratch
# ============================================================================

set -euo pipefail

echo "=== GDN Hybrid 7k v2 — 2×A100 Fallback Fresh Start ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date
nvidia-smi

source /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/.venv/bin/activate

# Per-job Triton cache
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"
export FLA_USE_NAIVE=0

# Model and seed
export ARCH_MODE=A
export SEED=${SEED:-42}
export RUN_ID="7k_v2_A_seed${SEED}_${SLURM_JOB_ID}"

# Data
export DATA_PATH=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/data/tokenizers/fineweb_1024_bpe.model

# Training config — 7000 steps with 30% warmdown
export ITERATIONS=7000
export WARMDOWN_ITERS=2100
export WARMUP_STEPS=20
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=1024
export EVAL_SEQ_LEN=1024
export MAX_WALLCLOCK_SECONDS=27000  # 7h30m safety

# NO RESUME — fresh start from scratch

# Validation and logging
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
export SAVE_EVERY=1000

# Optimizer
export MATRIX_LR=0.02
export SCALAR_LR=0.02
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=500
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3

# EMA + SWA
export EMA_DECAY=0.997
export SWA_ENABLED=1
export SWA_EVERY=50

# Late QAT
export LATE_QAT_THRESHOLD=0.15

# GPTQ post-processing
export GPTQ_ENABLED=1

# Eval
export EVAL_STRIDE=64
export XSA_EVAL=0
export COMPILE_ENABLED=1

# Checkpoint
export CKPT_DIR=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/checkpoints/7k_v2
mkdir -p "$CKPT_DIR"
mkdir -p /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/logs

# Run training — 2×A100 DDP
torchrun --standalone --nproc_per_node=2 train_gdn_7k.py 2>&1 | tee /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/logs/7k_v2_A_seed${SEED}_${SLURM_JOB_ID}.txt

echo "=== Done ==="
date
