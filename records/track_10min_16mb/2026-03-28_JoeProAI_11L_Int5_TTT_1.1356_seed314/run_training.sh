#!/bin/bash
# GlassBridge / JoeProAI — Parameter Golf submission runner
# Reproduces val_bpb=1.13256182 on 8xH100
#
# Usage:
#   bash run_training.sh
#
# Requirements:
#   - 8x NVIDIA H100 (80GB) GPUs
#   - Python 3.10+, CUDA 12.4+
#   - pip install -r requirements.txt
#   - Data: fineweb10B_sp1024 dataset at $DATA_PATH
#   - Tokenizer: fineweb_1024_bpe.model at $TOKENIZER_PATH

set -e

# ── Paths (edit these) ──────────────────────────────────────────────────────
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

# ── Training hyperparameters ─────────────────────────────────────────────────
export MATRIX_LR="0.025"
export SCALAR_LR="0.025"
export MUON_WD="0.0"
export ADAM_WD="0.0"
export GRAD_CLIP_NORM="0.0"
export MUON_MOMENTUM="0.95"
export WARMDOWN_ITERS="6000"

# ── TTT (Test-Time Training) config ──────────────────────────────────────────
export TTT_ENABLED="1"
export TTT_USE_ADAMW="1"
export TTT_ADAMW_LR="0.0004"
export TTT_ADAMW_WD="0.0"
export TTT_MLP_ONLY="1"
export TTT_EPOCHS="1"
export TTT_FREEZE_BLOCKS="0"

# ── Architecture ─────────────────────────────────────────────────────────────
export MLP_HIDDEN="1536"
export BIGRAM_BUCKETS="4096"
export PRUNE_PCT="0.15"

# ── Reproducibility ───────────────────────────────────────────────────────────
export SEED="314"

echo "Starting training run..."
echo "DATA_PATH: $DATA_PATH"
echo "TOKENIZER_PATH: $TOKENIZER_PATH"

DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
torchrun --nproc_per_node=8 train_gpt.py

echo "Training complete. Artifact: final_model.int5.ptz"
