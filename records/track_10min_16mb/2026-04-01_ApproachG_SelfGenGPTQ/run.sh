#!/bin/bash
# Approach G: AR Self-Gen GPTQ
# Run on 8xH100 SXM within 600s budget
set -euo pipefail

export NCCL_IB_DISABLE=1
export RUN_ID="${RUN_ID:-approach_g}"
export SEED="${SEED:-1337}"

# Model architecture
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=8
export VOCAB_SIZE=1024
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export BIGRAM_VOCAB_SIZE=6144
export BIGRAM_DIM=128
export XSA_LAST_N=11
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"

# Optimizer
export MUON_WD=0.04
export ADAM_WD=0.04
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3500

# Training budget: 390s train + ~210s AR self-gen GPTQ = 600s total
export MAX_WALLCLOCK_SECONDS=390
export ITERATIONS=20000

# Eval
export EVAL_STRIDE=64
export SWA_ENABLED=1

# Pruning
export PRUNE_PCT=0.10

# TTT
export TTT_EPOCHS=3
export TTT_LR=0.0001
export TTT_FREEZE_BLOCKS=2
export TTT_CHUNK_TOKENS=131072
export TTT_OPTIMIZER=adamw

torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "/workspace/run_${RUN_ID}.log"
