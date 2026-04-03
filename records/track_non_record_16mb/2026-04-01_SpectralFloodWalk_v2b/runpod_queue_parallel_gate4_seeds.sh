#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DIR}"

IFS=' ' read -r -a GPUS <<< "${SFW_QUEUE_GPUS:-0 1 2}"
IFS=' ' read -r -a SEEDS <<< "${SFW_QUEUE_SEEDS:-1337 42 2025}"
if (( ${#GPUS[@]} < ${#SEEDS[@]} )); then
  echo "[error] runpod_queue_parallel_gate4_seeds.sh needs at least one GPU id per queued seed." >&2
  exit 1
fi

declare -a pids=()

for idx in "${!SEEDS[@]}"; do
  gpu="${GPUS[$idx]}"
  seed="${SEEDS[$idx]}"
  echo "[queue] gpu=${gpu} script=runpod_gate4.sh seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  SFW_TARGET_HARDWARE="${SFW_TARGET_HARDWARE:-1xGPU queued}" \
  SFW_TARGET_GPU_COUNT=1 \
  SFW_NPROC_PER_NODE=1 \
  SFW_SEED="${seed}" \
  "${DIR}/runpod_gate4.sh" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

python3 ../../../tools/summarize_v2b_runs.py runs/*
