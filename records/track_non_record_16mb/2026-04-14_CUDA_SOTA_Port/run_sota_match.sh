#!/bin/bash
# Run CUDA training with SOTA architecture + tuned hyperparameters (from PR #1612).
# Expected: reproduce ~1.07 BPB on 8×H100 SXM with 2.4B tokens (524K batch × 4550 steps).

# Architecture features (matching SOTA, not upstream baseline)
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4

# Tokenizer (casefold SP8192 required — see records/ for training)
export VOCAB_SIZE=8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export DATA_PATH=./data/datasets/fineweb10B_sp8192

# SOTA hyperparameters (originally tuned for H100 large batch)
export QK_GAIN_INIT=5.25
export MATRIX_LR=0.022
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.95

# Architecture features
export RECUR_LAYERS=3,4,5
export RECUR_START_STEP=1500  # ~33% of 4550 steps
export PARALLEL_RESIDUAL=1
export PARALLEL_START_LAYER=7

# Training
export ITERATIONS=4550
export WARMDOWN_ITERS=3250  # 72% warmdown (H100 regime)
export TRAIN_BATCH_TOKENS=524288
export MAX_WALLCLOCK_SECONDS=600

torchrun --standalone --nproc_per_node=8 train_gpt.py
