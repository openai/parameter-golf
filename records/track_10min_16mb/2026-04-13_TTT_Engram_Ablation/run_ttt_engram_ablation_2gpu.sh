#!/bin/bash
set -euo pipefail

# ============================================================================
# TTT + Engram Ablation Runner — 2×GPU Competition Profile
# Trains once with engram enabled, then runs 4-case eval ablation:
#   1. baseline     (no TTT, no engram correction)
#   2. engram_only  (no TTT, with engram correction)
#   3. ttt_only     (TTT, no engram correction)
#   4. ttt_engram   (TTT, with engram correction)
# ============================================================================

# Core model config
export MODEL_DIM=600
export NUM_LAYERS=12
export NUM_HEADS=8
export NUM_KV_HEADS=2
export TIE_EMBEDDINGS=1
export FP_STORAGE=fp4

# Engram config (sized for 24GB GPUs)
export BIGRAM_HASH_ENABLED=1
export ENGRAM_COMPETITION_ENABLED=1
export BIGRAM_HASH_BUCKETS=${BIGRAM_HASH_BUCKETS:-8192}
export BIGRAM_HASH_DIM=${BIGRAM_HASH_DIM:-64}
export ENGRAM_NUM_HEADS=${ENGRAM_NUM_HEADS:-2}
export ENGRAM_NUM_ORDERS=${ENGRAM_NUM_ORDERS:-2}
export ENGRAM_EXPORT_TOKEN_BUDGET=${ENGRAM_EXPORT_TOKEN_BUDGET:-8192}

# Batch size (fit in 24GB VRAM — 2 GPUs, each gets half)
export TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-16384}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16384}
export SLIDING_BATCH_SIZE=${SLIDING_BATCH_SIZE:-64}

# NCCL fix for PCIe-connected 3090s (P2P hangs on some pods)
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

# DDP: SKC competition has unused params (feedback/capsule/attention disabled but params exist)
export DDP_FIND_UNUSED_PARAMETERS=1

# Compile mode (skip compilation for 10-min runs — overhead > benefit)
export COMPILE_MODE=${COMPILE_MODE:-none}

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable budget enforcement during training (verbose source is larger than compressed submission)
export HARD_BUDGET_ENFORCE=0

# TTT config
export TTT_ENABLED=1
export TTT_SCOPE=skc_safe
export TTT_LR=${TTT_LR:-0.005}
export TTT_EPOCHS=${TTT_EPOCHS:-3}
export TTT_CHUNK_TOKENS=${TTT_CHUNK_TOKENS:-32768}
export TTT_MOMENTUM=${TTT_MOMENTUM:-0.9}

# Ablation flag — triggers all 4 eval cases after training
export ABLATION_EVAL=1

# Sliding eval
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE=${SLIDING_EVAL_STRIDE:-64}

# Distributed & competition
export NPROC=2
export ARCHITECTURE=skc_competition
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}
export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_8192_bpe.model}"
export VOCAB_SIZE=8192

DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${DIR}/../../.." && pwd)}"
TRAINER_PATH="${TRAINER_PATH:-train_gpt_verbose.py}"

export RUN_ID="ttt_engram_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="${DIR}/logs/${RUN_ID}.log"

echo "=========================================================================="
echo "  TTT + ENGRAM ABLATION — ${NPROC}×GPU"
echo "  ENGRAM: buckets=${BIGRAM_HASH_BUCKETS} dim=${BIGRAM_HASH_DIM} heads=${ENGRAM_NUM_HEADS} orders=${ENGRAM_NUM_ORDERS}"
echo "  TTT: scope=${TTT_SCOPE} lr=${TTT_LR} epochs=${TTT_EPOCHS}"
echo "  ABLATION_EVAL=1 (4-case eval)"
echo "=========================================================================="

OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=${NPROC} "${PROJECT_ROOT}/${TRAINER_PATH}" 2>&1 | tee "${LOG}"

echo "=== DONE ==="
