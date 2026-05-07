#!/bin/bash
#SBATCH --job-name=varC_s42
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --nice=0
#SBATCH --output=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/experiments_16mb/varC_11L_int5mix/runs/seed42_%j/logs/train_%j.out
#SBATCH --error=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/experiments_16mb/varC_11L_int5mix/runs/seed42_%j/logs/train_%j.err
#SBATCH --account=medcam

# =============================================================================
# Variant C: 11L + BigramHash(1536) + Long Warmdown (60%) + int5 MLP — Seed 42
# Goal: int5 quantization for MLP weights → ~1 bit/param savings under compression
# =============================================================================
set -euo pipefail

VARIANT_DIR=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/experiments_16mb/varC_11L_int5mix
RUN_DIR=${VARIANT_DIR}/runs/seed42_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/checkpoints"

echo "=== Variant C: 11L + int5 MLP — Seed 42 ==="
echo "Run dir: ${RUN_DIR}"
echo "Host: $(hostname), GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
date

source /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/.venv/bin/activate
cd "${RUN_DIR}"

NGPU=2
export MASTER_PORT=$((10000 + RANDOM % 50000))

export RUN_ID="varC_seed42_${SLURM_JOB_ID}"
export SEED=42

export DATA_PATH=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/repo/data/datasets/fineweb10B_sp1024/
export TOKENIZER_PATH=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/repo/data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024

export MAX_WALLCLOCK_SECONDS=0
export ITERATIONS=7185
export WARMDOWN_ITERS=4311
export WARMUP_STEPS=20
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=200
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048

export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export TIE_EMBEDDINGS=1
export BIGRAM_VOCAB_SIZE=1536
export BIGRAM_DIM=128
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export LOGIT_SOFTCAP=30.0
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"

# --- E2E TTT: ENABLED ---
export E2E_TTT_ENABLED=1
export E2E_TTT_NUM_HEADS=8
export E2E_TTT_MINI_BATCH=16
export E2E_TTT_BASE_LR=1.0

# --- FlowRefiner: ENABLED ---
export FLOW_ENABLED=1
export FLOW_LATENT_DIM=64
export FLOW_HIDDEN_DIM=256
export FLOW_INIT_SCALE=0.01

# --- int5 MLP quantization ---
export INT5_MLP=1

# --- Checkpointing ---
export SAVE_EVERY=1000
export SAVE_DIR=${RUN_DIR}/checkpoints

export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3
export EVAL_STRIDE=64

echo "Config: SEED=$SEED ITERATIONS=$ITERATIONS WARMDOWN=$WARMDOWN_ITERS BIGRAM=$BIGRAM_VOCAB_SIZE NUM_LAYERS=$NUM_LAYERS INT5_MLP=$INT5_MLP NGPU=$NGPU"

torchrun --standalone --nproc_per_node=$NGPU \
    "${VARIANT_DIR}/train_gpt.py" 2>&1 | \
    tee "${RUN_DIR}/logs/train_${SLURM_JOB_ID}.txt"

echo "=== EXIT: $? ==="
date
