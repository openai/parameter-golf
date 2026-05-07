#!/bin/bash
# Approach E: SLOT + QK-Gain(4.0) + Int5 GPTQ + score-first TTT
# Environment: 8xH100 SXM, PyTorch 2.9+, flash-attn
set -euo pipefail

export NCCL_IB_DISABLE=1

# Model config (same as Approach B base)
export NUM_LAYERS=11
export NUM_KV_HEADS=8
export MODEL_DIM=512
export NUM_HEADS=8
export MLP_MULT=3.5
export BIGRAM_VOCAB_SIZE=6144
export BIGRAM_DIM=128
export XSA_LAST_N=11
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048

# QK-Gain: raised from 1.5 to 4.0
export QK_GAIN_INIT=4.0

# Optimizer config
export MUON_WD=0.04
export ADAM_WD=0.04
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3500
export MAX_WALLCLOCK_SECONDS=590
export EVAL_STRIDE=64

# Late QAT
export LATE_QAT_THRESHOLD=0.5

# Pruning
export PRUNE_PCT=0.10

# TTT config (score-first)
export TTT_EPOCHS=3
export TTT_LR=0.0001
export TTT_FREEZE_BLOCKS=2
export TTT_CHUNK_TOKENS=131072
export TTT_OPTIMIZER=adamw

# SLOT config (per-batch delta optimization)
export SLOT_ENABLED=1
export SLOT_STEPS=8
export SLOT_LR=0.005

torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/run.log
