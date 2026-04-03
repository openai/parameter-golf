#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/flash-attention/hopper${PYTHONPATH:+:${PYTHONPATH}}"
export RUN_ID="${RUN_ID:-slot_fix_${SEED:-300}_$(date +%Y%m%d_%H%M%S)}"
export SEED="${SEED:-300}"
export SLOT_ENABLED="${SLOT_ENABLED:-1}"
export SLOT_STEPS="${SLOT_STEPS:-1}"
export SLOT_LR="${SLOT_LR:-1e-2}"
export SLOT_POWER="${SLOT_POWER:-0.30}"
export ITERATIONS="${ITERATIONS:-64}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-16}"
export POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}"
export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
export EXIT_AFTER_SIZE_ONLY="${EXIT_AFTER_SIZE_ONLY:-1}"
export COMPILE_ENABLED="${COMPILE_ENABLED:-0}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)}"
if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" == "0" ]]; then
  NPROC_PER_NODE=1
fi

if [ -z "${SKIP_GPTQ:-}" ]; then
  export SKIP_GPTQ=1
fi

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  experiments/slot_fix_spark/train_gpt.py
