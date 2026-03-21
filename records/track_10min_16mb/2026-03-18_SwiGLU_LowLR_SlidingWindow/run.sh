#!/bin/bash
# SwiGLU + Lower LR + Sliding Window Eval submission
# Run on 8xH100 via Runpod with the Parameter Golf template

set -euo pipefail

# Dataset (should already be downloaded in Runpod template)
DATA_PATH=${DATA_PATH:-./data/datasets/fineweb10B_sp1024}
TOKENIZER_PATH=${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}

# Download data if not present
if [ ! -d "$DATA_PATH" ]; then
    echo "Downloading dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

# Wandb setup (optional)
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="parameter-golf"

# Run training with our improvements
NCCL_IB_DISABLE=1 \
RUN_ID="submission_swiglu_lowlr_sw_$(date +%s)" \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
USE_SWIGLU=1 \
SWIGLU_HIDDEN=672 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
LOGIT_SOFTCAP=30 \
ADAM_EPS=1e-8 \
EVAL_STRIDE=256 \
EVAL_SEQ_LEN=0 \
LAWA_ENABLED=1 \
LAWA_INTERVAL=100 \
BYTE_GROUPING=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo "Done! Check logs/ directory for training log."
echo "Check final_model.int8.ptz for the compressed artifact."
echo "Look for 'sliding_window_eval' lines in the log for the final BPB."
