#!/bin/bash
#SBATCH --job-name=gdn_eval_7k_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/logs/eval_7k_v2_%j.out
#SBATCH --error=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/logs/eval_7k_v2_%j.err
#SBATCH --chdir=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments

# ============================================================================
# Eval: GDN PureGDN 7k v2 — Sliding-window + Legal TTT
# Runs BOTH evaluations on the final artifact:
#   Phase 1: Plain sliding-window (no TTT)
#   Phase 2: Legal score-first TTT
# ============================================================================

set -euo pipefail

echo "=== GDN Hybrid 7k v2 — Evaluation ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date
nvidia-smi

source /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/.venv/bin/activate

# Per-job Triton cache
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"
export FLA_USE_NAIVE=0
export PYTHONUNBUFFERED=1

mkdir -p /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/logs

# ─── Artifact path (override via: ARTIFACT_PATH=... sbatch slurm_eval_7k_v2.sh) ───
export ARTIFACT_PATH=${ARTIFACT_PATH:-/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/checkpoints/7k_v2/final_model_A_PureGDN_seed42.int6.ptz}

if [ ! -f "$ARTIFACT_PATH" ]; then
    echo "ERROR: Artifact not found at $ARTIFACT_PATH"
    echo "Set ARTIFACT_PATH env var to the correct .int6.ptz or .pt file"
    exit 1
fi

echo "Artifact: $ARTIFACT_PATH"
echo "Artifact size: $(stat --format=%s "$ARTIFACT_PATH") bytes"

# Model config
export ARCH_MODE=A

# Data
export DATA_PATH=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/data/tokenizers/fineweb_1024_bpe.model

# Eval settings
export EVAL_SEQ_LEN=1024
export EVAL_STRIDE=64

# TTT settings — eval_ttt.py runs BOTH phases internally when TTT_ENABLED=1
export TTT_ENABLED=1
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=2
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

# Run evaluation (eval_ttt.py runs both plain + TTT phases)
python eval_ttt.py 2>&1 | tee /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/logs/eval_7k_v2_${SLURM_JOB_ID}.txt

echo "=== Eval Done ==="
date
