#!/bin/bash
# Experiment: ptq=int6attn+int5mlp L=20 mlp=5 MTP k=4
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/../../.env" ] && source "${SCRIPT_DIR}/../../.env" || { [ -f .env ] && source .env; }
export COMET_API_KEY="${COMET_API_KEY:-}"

export RUN_ID="ptq_int5mlp_L20_mlp5_mtp4"
export EXPERIMENT_NAME="ptq=int6attn+int5mlp L=20 mlp=5 MTP=4"

# Model architecture
export NUM_LAYERS=20
export MODEL_DIM=256
export MLP_MULT=5
export NUM_HEADS=8
export NUM_KV_HEADS=4
export VOCAB_SIZE=1024
export TIE_EMBEDDINGS=1
export ROPE_BASE=10000.0
export LOGIT_SOFTCAP=30.0
export QK_GAIN_INIT=1.5

# Quantization
export TERNARY_ENABLED=0
export QAT_BITS=0
export PTQ_BITS=6
export PTQ_MLP_BITS=5

# Training
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export ITERATIONS=4500
export MAX_WALLCLOCK_SECONDS=600
export WARMUP_STEPS=20
export WARMDOWN_ITERS=1200
export SEED=1337

export OPTIMIZER=muon_adam
export LR_SCHEDULE=trapezoid
export EMBED_LR=0.6
export HEAD_LR=0.008
export TIED_EMBED_LR=0.05
export TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.04
export SCALAR_LR=0.04
export MUON_MOMENTUM=0.95
export MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=500
export BETA1=0.9
export BETA2=0.95
export ADAM_EPS=1e-8
export ADAMW_WEIGHT_DECAY=0.1
export GRAD_CLIP_NORM=0.0

# Multi-Token Prediction (k=4: predict 4 tokens ahead, keep only head 0 at inference)
export MTP_NUM_HEADS=4
export MTP_LOSS_WEIGHT=0.3

# MHC
export USE_MHC=0
export MHC_TYPE=mhc
export MHC_NUM_STREAMS=2

# Compile
export USE_COMPILE=1
export USE_COMPILE_MODEL=1
export USE_COMPILE_MUON=1
export ALLOW_COMPILE_WITH_MHC=0
export COMPILE_FULLGRAPH=1
export COMPILE_DYNAMIC=0

# FlashAttention — target wants version=3, but flash-attn not installed (no nvcc).
# Falls back to torch SDPA.
export USE_FLASHATTENTION2=0
export USE_FLASHATTENTION3=0

# Comet (for submission we turn off comet logging)
export COMET_ENABLE=0
export COMET_PROJECT_NAME=parameter-golf
export COMET_LOG_TRAIN_EVERY=100

# Validation
export VAL_BATCH_SIZE=524288
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=50

python3 -m torch.distributed.run --nproc_per_node=8 train_gpt.py
