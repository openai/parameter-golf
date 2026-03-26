#!/bin/bash
# Eval-only: loads saved model, runs one eval method variant
# Usage: GPU=0 METHOD=baseline WANDB_RUN_NAME=round1_gpu0_baseline bash eval_only.sh

export PATH=/data/backups/rganapa/pylibs/bin:$PATH
export PYTHONPATH=/data/backups/rganapa/pylibs
export TMPDIR=/data/backups/rganapa/tmp
export TRITON_CACHE_DIR=/data/backups/rganapa/triton_cache
export TORCH_HOME=/data/backups/rganapa/torch_home
export DATA_PATH=data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=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
export WANDB_PROJECT=parameter-golf

# Use TTT_ONLY mode to skip training and just eval
export TTT_ONLY=1
export TTT_CHUNK_TOKENS=${TTT_CHUNK_TOKENS:-1048576}
export TTT_EPOCHS=${TTT_EPOCHS:-4}
export TTT_LR=${TTT_LR:-0.0005}
export TTT_FREEZE_BLOCKS=${TTT_FREEZE_BLOCKS:-2}
export SEED=${SEED:-1337}

cd /data/backups/rganapa/parameter-golf

CUDA_VISIBLE_DEVICES=${GPU:-0} \
  python3 pr834_train_gpt.py
