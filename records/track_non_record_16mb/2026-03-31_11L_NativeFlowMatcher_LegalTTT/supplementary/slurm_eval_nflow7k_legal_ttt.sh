#!/bin/bash
#############################################################################
# Eval: NativeFlowMatcher 7k checkpoint with LEGAL single-pass TTT
# Model: final_model_pr940_nflow_55342820.pt (27,530,952 params)
# Baseline (no TTT): sliding BPB = 1.12312
# Expected runtime: ~2h (legal TTT interleaved with sliding window)
#############################################################################
#SBATCH --job-name=eval_nflow7k_lttt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=medcam
#SBATCH --output=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/logs/eval_nflow7k_legal_ttt_%j.out
#SBATCH --error=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/logs/eval_nflow7k_legal_ttt_%j.err
#SBATCH --chdir=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────
source /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

mkdir -p /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/logs

# ── Run identifiers ─────────────────────────────────────────────────────
export RUN_ID="eval_nflow7k_legal_ttt_${SLURM_JOB_ID}"
LOGFILE="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/logs/eval_nflow7k_legal_ttt_${SLURM_JOB_ID}.txt"

# ── Eval-only: load checkpoint ──────────────────────────────────────────
export EVAL_ONLY="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/runs/nflow_55342820/models/final_model_pr940_nflow_55342820.pt"

# ── Data paths ───────────────────────────────────────────────────────────
export DATA_PATH="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/repo/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/repo/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024

# ── Architecture (must match slurm_pr940_nflow_7k.sh exactly) ───────────
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

# ── NativeFlowMatcher (must match nflow training config) ────────────────
export NATIVE_FLOW_ENABLED=1
export NATIVE_FLOW_HIDDEN_DIM=256
export NATIVE_FLOW_INIT_SCALE=0.01
export NATIVE_FLOW_LOSS_WEIGHT=0.1

# ── Disable other modules ───────────────────────────────────────────────
export FLOW_ENABLED=0
export E2E_TTT_ENABLED=0
export EMA_ENABLED=0
export SWA_ENABLED=0
export QAT_ENABLED=0
export CANON_LAST_N=0

# ── Legal TTT config ────────────────────────────────────────────────────
export TTT_ENABLED=1
export LEGAL_TTT=1
export TTT_OPTIMIZER=sgd
export TTT_LR=0.002
export TTT_EPOCHS=10
export TTT_FREEZE_BLOCKS=2
export TTT_BATCH_SEQS=32
export TTT_CHUNK_TOKENS=32768
export TTT_GRAD_CLIP=1.0
export TTT_MOMENTUM=0.9

# ── Eval config ──────────────────────────────────────────────────────────
export EVAL_STRIDE=64
export SEED=42

# ── Training params (unused but required by argparse) ────────────────────
export ITERATIONS=7000
export WARMDOWN_ITERS=2800
export WARMUP_STEPS=20
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048

# ── Optimizer params (unused but required) ───────────────────────────────
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3

# ── Run ──────────────────────────────────────────────────────────────────
echo "=== Eval NativeFlow 7k with Legal TTT ==="
echo "Checkpoint: ${EVAL_ONLY}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"

torchrun --standalone --nproc_per_node=1 train_gpt_pr940.py 2>&1 | tee "$LOGFILE"

echo "End: $(date)"
