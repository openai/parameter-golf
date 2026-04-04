#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export COMPILE_MODEL="${COMPILE_MODEL:-1}"
export COMPILE_MUON="${COMPILE_MUON:-1}"
export ADAM_FUSED="${ADAM_FUSED:-1}"
export USE_LIBUV="${USE_LIBUV:-0}"
export EVAL_ONLY="${EVAL_ONLY:-1}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-final_model.pt}"
export EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-580}"
export EVAL_STRIDE="${EVAL_STRIDE:-256}"
export EVAL_BATCH_SEQS="${EVAL_BATCH_SEQS:-32}"
export TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-32}"
export TTT_TRAIN_BATCH_SEQS="${TTT_TRAIN_BATCH_SEQS:-8}"
export TTT_ENABLED="${TTT_ENABLED:-0}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train_gpt.py
