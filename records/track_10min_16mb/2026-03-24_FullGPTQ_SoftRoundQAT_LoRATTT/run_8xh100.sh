#!/bin/bash
set -euo pipefail

# Full 8xH100 training + evaluation run
# Usage: bash run_8xh100.sh [seed]
SEED=${1:-42}

export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=2
export MLP_MULT=3
export TIE_EMBEDDINGS=1
export TIED_EMBED_INIT_STD=0.02
export LOGIT_SOFTCAP=30.0
export ROPE_BASE=10000
export QK_GAIN_INIT=1.0
export LN_SCALE=1
export XSA_LAST_N=4
export ROPE_DIMS=16
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=128
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export GATED_ATTENTION=0
export VALUE_RESIDUAL=1

# Training schedule
export ITERATIONS=4600
export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_SIZE=64
export WARMUP_ITERS=200
export WARMDOWN_FRACTION=0.26
export LEARNING_RATE=0.0036
export MUON_LR=0.0126
export WEIGHT_DECAY=0.0
export ADAM_BETA2=0.95
export MIN_LR_RATIO=0.0
export EMA_DECAY=0.95
export EMA_START_STEP=800

# Late QAT (soft-round)
export LATE_QAT_THRESHOLD=0.18

# TTT (LoRA mode)
export TTT_ENABLED=1
export TTT_LORA=1
export TTT_LORA_RANK=8
export TTT_LORA_LR=0.005
export TTT_LORA_ALPHA=16.0
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=2
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

export RANDOM_SEED=$SEED
export MAX_WALLCLOCK_MS=540000

echo "=== Full GPTQ + Soft-Round QAT + LoRA TTT run (seed=$SEED) ==="
torchrun --nproc_per_node=8 train_gpt.py

echo "=== Done ==="
