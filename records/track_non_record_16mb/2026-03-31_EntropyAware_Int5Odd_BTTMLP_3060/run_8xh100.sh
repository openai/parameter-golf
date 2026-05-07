#!/usr/bin/env bash
set -euo pipefail

# Single-node 8xH100 launch template.
# Override any variable below from the shell as needed.

: "${NPROC_PER_NODE:=8}"
: "${RUN_ID:=cloud_8xh100}"
: "${TRAIN_BATCH_TOKENS:=131072}"
: "${TRAIN_MICROBATCH_TOKENS:=8192}"
: "${VAL_BATCH_SIZE:=1048576}"
: "${VAL_TOKEN_LIMIT:=0}"
: "${LR_SCALE_METHOD:=sqrt}"
: "${LR_REFERENCE_BATCH_TOKENS:=16384}"
: "${WARMUP_SCALE_METHOD:=linear}"
: "${WARMUP_REFERENCE_BATCH_TOKENS:=16384}"
: "${WARMUP_STEPS:=1}"
: "${COMPILE_STRUCTURED_MLP:=1}"
: "${ENABLE_COMPILE:=0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_gpt.py
