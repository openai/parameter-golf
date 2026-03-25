#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Causal JEPA on 1×A100/H100 — configurable duration (default 10 min)
#
# Google Colab setup:
#   !git clone -b trident-neural-memory-ttt https://github.com/<you>/parameter-golf.git
#   %cd parameter-golf
#   !pip install zstandard
#   # Download byte260 data:
#   !mkdir -p data/byte260_export logs
#   !cat > data/tokenizer_specs_byte260.json << 'SPEC'
#   [{"kind":"byte","name":"pure_byte_260","dataset_suffix":"byte260"}]
#   SPEC
#   !python data/download_hf_docs_and_tokenize.py \
#       --output-root data/byte260_export \
#       --tokenizer-config data/tokenizer_specs_byte260.json
#   !PYTHONUNBUFFERED=1 bash run_a100_causal_jepa.sh 2>&1 | tee logs/causal_jepa.txt
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

export DATA_PATH="${DATA_PATH:-./data/byte260_export/datasets/fineweb10B_byte260}"

# Timing
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export ITERATIONS="${ITERATIONS:-200000}"
export WARMUP_STEPS="${WARMUP_STEPS:-10}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-200}"

# Sequence & batching
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

# Architecture
export VOCAB_SIZE="${VOCAB_SIZE:-260}"
export MODEL_DIM="${MODEL_DIM:-480}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-3}"
export PARTIAL_ROPE_DIM="${PARTIAL_ROPE_DIM:-16}"
export ROPE_BASE="${ROPE_BASE:-10000}"
export LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-30.0}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
export BIGRAM_DIM="${BIGRAM_DIM:-128}"

# JEPA auxiliary
export JEPA_WEIGHT="${JEPA_WEIGHT:-0.1}"
export JEPA_LATENT_DIM="${JEPA_LATENT_DIM:-256}"
export JEPA_HORIZON="${JEPA_HORIZON:-16}"
export EMA_DECAY="${EMA_DECAY:-0.997}"

# Optimizer
export EMBED_LR="${EMBED_LR:-0.6}"
export MATRIX_LR="${MATRIX_LR:-0.025}"
export SCALAR_LR="${SCALAR_LR:-0.04}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.3}"

# Quantization
export INT6_ENABLED="${INT6_ENABLED:-1}"
export USE_ZSTD="${USE_ZSTD:-1}"
export ZSTD_LEVEL="${ZSTD_LEVEL:-22}"

# Evaluation
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-131072}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export EVAL_BATCH_SEQS="${EVAL_BATCH_SEQS:-16}"

# TTT
export TTT_ENABLED="${TTT_ENABLED:-1}"
export TTT_LR="${TTT_LR:-0.002}"
export TTT_EPOCHS="${TTT_EPOCHS:-3}"
export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
export TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"
export TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"

export RUN_ID="${RUN_ID:-causal_jepa_a100}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"

python ./jepa/train_gpt.py
