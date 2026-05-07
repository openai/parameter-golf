#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"

if [[ ! -f "${repo_root}/.venv-wsl/bin/activate" ]]; then
  echo "Missing WSL venv at ${repo_root}/.venv-wsl" >&2
  echo "Set up the environment first." >&2
  exit 1
fi

source "${repo_root}/.venv-wsl/bin/activate"
cd "${repo_root}"

if ! python -c "import zstandard" >/dev/null 2>&1; then
  pip install zstandard
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export RUN_ID="${RUN_ID:-h100_8x_no_novel_606ish}"
export SEED="${SEED:-1337}"

export ITERATIONS="${ITERATIONS:-200000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-597}"
export WARMUP_STEPS="${WARMUP_STEPS:-3}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1200}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export ENABLE_FINAL_EVAL="${ENABLE_FINAL_EVAL:-0}"
export MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-131072}"

export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-1}"
export SDP_BACKEND="${SDP_BACKEND:-flash}"
export USE_FA3="${USE_FA3:-1}"

export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export VAL_CONTEXT_LEN="${VAL_CONTEXT_LEN:-2048}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
export EVAL_BATCH_SEQS="${EVAL_BATCH_SEQS:-16}"
export EVAL_STRIDE="${EVAL_STRIDE:-32}"

export NUM_LAYERS="${NUM_LAYERS:-10}"
export MODEL_DIM="${MODEL_DIM:-480}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-8}"
export MLP_MULT="${MLP_MULT:-3.5}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-1.2}"

export PARTIAL_ROPE_DIM="${PARTIAL_ROPE_DIM:-16}"
export LN_SCALE_ENABLED="${LN_SCALE_ENABLED:-1}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
export BIGRAM_DIM="${BIGRAM_DIM:-128}"
export XSA_ENABLED="${XSA_ENABLED:-1}"
export XSA_TOP_LAYERS="${XSA_TOP_LAYERS:-11}"
export XSA_SCALE="${XSA_SCALE:-1.0}"
export XSA_GATED="${XSA_GATED:-1}"

export D_MEMORY="${D_MEMORY:-0}"
export TRUNK_D_MEMORY="${TRUNK_D_MEMORY:-0}"
export CORE_D_MEMORY="${CORE_D_MEMORY:-0}"
export CORE_MEMORY_ONLY="${CORE_MEMORY_ONLY:-0}"
export LOOP_CORE_ENABLED="${LOOP_CORE_ENABLED:-0}"
export LOOP_CORE_LAYERS="${LOOP_CORE_LAYERS:-0}"
export LOOP_REPEATS="${LOOP_REPEATS:-1}"
export VALUE_RESIDUAL="${VALUE_RESIDUAL:-0}"
export GATED_ATTENTION="${GATED_ATTENTION:-0}"
export CATALYTIC_RESIDUAL="${CATALYTIC_RESIDUAL:-0}"
export POST_TRAINING_EVAL_MODE="${POST_TRAINING_EVAL_MODE:-none}"

export QAT_ENABLED="${QAT_ENABLED:-1}"
export QAT_START_FRAC="${QAT_START_FRAC:-0.75}"
export SWA_ENABLED="${SWA_ENABLED:-0}"
export SWA_START_FRAC="${SWA_START_FRAC:-0.5}"
export SWA_EVERY="${SWA_EVERY:-50}"
export EMA_ENABLED="${EMA_ENABLED:-1}"
export EMA_START_FRAC="${EMA_START_FRAC:-0.85}"
export EMA_DECAY="${EMA_DECAY:-0.997}"
export SOFT_ROUND_QAT="${SOFT_ROUND_QAT:-1}"
export SOFT_ROUND_ALPHA_START="${SOFT_ROUND_ALPHA_START:-1}"
export SOFT_ROUND_ALPHA_END="${SOFT_ROUND_ALPHA_END:-16}"
export PRUNE_PCT="${PRUNE_PCT:-0.02}"

export INT5_MLP_EXPORT="${INT5_MLP_EXPORT:-1}"
export INT6_ATTN_EXPORT="${INT6_ATTN_EXPORT:-1}"
export INT6_OTHER_EXPORT="${INT6_OTHER_EXPORT:-0}"
export DISABLE_QUANTIZATION="${DISABLE_QUANTIZATION:-0}"
export USE_ZSTD="${USE_ZSTD:-1}"
export ZSTD_LEVEL="${ZSTD_LEVEL:-22}"

export MATRIX_LR="${MATRIX_LR:-0.025}"
export MUON_WD="${MUON_WD:-0.04}"

torchrun --standalone --nproc_per_node=8 ./train_gpt.py
