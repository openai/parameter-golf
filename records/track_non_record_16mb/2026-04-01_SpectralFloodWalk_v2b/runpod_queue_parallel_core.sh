#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DIR}"

IFS=' ' read -r -a GPUS <<< "${SFW_QUEUE_GPUS:-0 1 2}"
if (( ${#GPUS[@]} < 3 )); then
  echo "[error] runpod_queue_parallel_core.sh needs at least 3 GPU ids in SFW_QUEUE_GPUS." >&2
  exit 1
fi

declare -a pids=()
declare -a labels=()

launch_one() {
  local gpu="$1"
  shift
  local script="$1"
  shift
  echo "[queue] gpu=${gpu} script=${script}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  SFW_TARGET_HARDWARE="${SFW_TARGET_HARDWARE:-1xGPU queued}" \
  SFW_TARGET_GPU_COUNT=1 \
  SFW_NPROC_PER_NODE=1 \
  "${DIR}/${script}" "$@" &
  pids+=("$!")
  labels+=("${script}")
}

launch_one "${GPUS[0]}" runpod_baseline.sh
launch_one "${GPUS[1]}" runpod_gate.sh
launch_one "${GPUS[2]}" runpod_gate4.sh

queue_status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  label="${labels[$idx]}"
  if ! wait "${pid}"; then
    echo "[error] queued job failed: ${label} (pid=${pid})" >&2
    queue_status=1
  fi
done

python3 ../../../tools/summarize_v2b_runs.py runs/*
exit "${queue_status}"
