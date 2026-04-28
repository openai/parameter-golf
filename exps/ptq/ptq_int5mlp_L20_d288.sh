#!/bin/bash
# PTQ int6 attn + int5 MLP, L=20, model_dim=288. User-specified hparams + remainder from ptq_int5mlp_mlp7.sh.
# Note: ptq_layer_{start,end} below map to INT6_LAYER_{START,END} (serialization int6 layer-range override).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/../../.env" ]; then
  source "${SCRIPT_DIR}/../../.env"
elif [ -f .env ]; then
  source .env
fi

export RUN_ID="ptq_int5mlp_L20_d288"
export EXPERIMENT_NAME="ptq=int6attn+int5mlp L=20 d=288 seq_len=512"

# --- User-specified (see also Hyperparameters / env names in train_gpt_lib/config.py) ---
export INT6_LAYER_START=-1
export INT6_LAYER_END=-1
export PTQ_MLP_BITS=5
export QAT_BITS=0
export QAT_MLP_BITS=0
# Comet activation Frobenius norms (1=on, 0=off; cadence ACTIVATION_NORM_LOG_EVERY).
export LOG_ACTIVATION_NORMS=0
export QK_GAIN_INIT=1.5
export SCALAR_LR=0.04
export SEED=1337
export TIE_EMBEDDINGS=1
export TIED_EMBED_INIT_STD=0.005
export TIED_EMBED_LR=0.05
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export TRAIN_BATCH_TOKENS=524288
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TRAIN_LOG_EVERY=50
export TRAIN_SEQ_LEN=512
export USE_COMPILE=1
export USE_COMPILE_MODEL=1
export USE_COMPILE_MUON=1
export USE_MHC=0
export VAL_BATCH_SIZE=524288
export VAL_LOSS_EVERY=1000
export VOCAB_SIZE=1024
export WARMDOWN_ITERS=1200
export WARMUP_STEPS=20

# Model architecture (same as ptq_int5mlp_mlp7.sh except MODEL_DIM for d=288)
export NUM_LAYERS=20
export MODEL_DIM=288
export MLP_MULT=5
export NUM_HEADS=8
export NUM_KV_HEADS=4
export ROPE_BASE=10000.0
export LOGIT_SOFTCAP=30.0

# Quantization
export TERNARY_ENABLED=0
export PTQ_BITS=6

# Training (unchanged from ptq_int5mlp_mlp7.sh except values duplicated above)
export ITERATIONS=4500
export MAX_WALLCLOCK_SECONDS=600

export OPTIMIZER=muon_adam
export LR_SCHEDULE=trapezoid
export EMBED_LR=0.6
export HEAD_LR=0.008
export MATRIX_LR=0.04
export MUON_MOMENTUM=0.95
export MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=500
export BETA1=0.9
export BETA2=0.95
export ADAM_EPS=1e-8
export ADAMW_WEIGHT_DECAY=0.1
export GRAD_CLIP_NORM=0.0

# Multi-Token Prediction
export MTP_NUM_HEADS=1
export MTP_LOSS_WEIGHT=1.0

# MHC
export MHC_TYPE=mhc
export MHC_NUM_STREAMS=2

# Compile
export ALLOW_COMPILE_WITH_MHC=0
export COMPILE_FULLGRAPH=1
export COMPILE_DYNAMIC=0

# FlashAttention — falls back to torch SDPA if flash-attn not installed.
export USE_FLASHATTENTION2=0
export USE_FLASHATTENTION3=0

# Comet
export COMET_ENABLE=1
export COMET_API_KEY="wKvWIXBmWdm5O9w8buIWrqKEV"
export COMET_PROJECT_NAME=parameter-golf
export COMET_LOG_TRAIN_EVERY=100

python3 -m torch.distributed.run --nproc_per_node=8 train_gpt.py
