#!/bin/bash
# Run CUDA training with OUR TUNED hyperparameters from PR #1612.
# This is the hypothesis: our tuned values (Muon 0.95, QK 4.0, LR 0.02) beat SOTA defaults at H100 scale.

# Architecture (same as SOTA)
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4

# Tokenizer
export VOCAB_SIZE=8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export DATA_PATH=./data/datasets/fineweb10B_sp8192

# TUNED hyperparameters (PR #1612 winning config)
export QK_GAIN_INIT=4.0
export MATRIX_LR=0.02
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.90

# Architecture features
export RECUR_LAYERS=3,4,5
export RECUR_START_STEP=1500
export PARALLEL_RESIDUAL=1
export PARALLEL_START_LAYER=7

# Training
export ITERATIONS=4550
export WARMDOWN_ITERS=3250
export TRAIN_BATCH_TOKENS=524288
export MAX_WALLCLOCK_SECONDS=600

torchrun --standalone --nproc_per_node=8 train_gpt.py
