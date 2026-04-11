#!/bin/bash
# Experiment: ptq=int6attn+int5mlp L=16 mlp=7
# This was identified as one of the best performing configurations.

export RUN_ID="ptq_int6attn_int5mlp_L16_mlp7"
export EXPERIMENT_NAME="PTQ int6-attn int5-mlp L=16 mlp=7"

# Architecture
export NUM_LAYERS=16
export MODEL_DIM=256
export MLP_MULT=7

# Quantization
export TERNARY_ENABLED=0
export QAT_BITS=0
export PTQ_BITS=6
export PTQ_MLP_BITS=5

# Training
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600

# Optimizer
export OPTIMIZER=muon_adam
export LR_SCHEDULE=trapezoid

torchrun --nproc_per_node=8 train_gpt.py
