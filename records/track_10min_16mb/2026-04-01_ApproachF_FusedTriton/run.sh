#!/bin/bash
# Approach F: Fused Triton MLP kernel for faster training
# Run on 8xH100 SXM with RunPod
set -euo pipefail

cd /workspace/parameter-golf

# Ensure data is available
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

export NCCL_IB_DISABLE=1

# Main training run
NCCL_IB_DISABLE=1 RUN_ID=approach_f_fused_triton \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=8 \
MLP_MULT=3.5 \
BIGRAM_VOCAB_SIZE=6144 \
BIGRAM_DIM=128 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=590 \
TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
PRUNE_PCT=0.10 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/run.log
