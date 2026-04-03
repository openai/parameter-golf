#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/records/track_non_record_16mb/2026-04-03_DeepFloor/runs/matrix}"
ENWIK8_PATH="${ENWIK8_PATH:-}"
DEVICE="${DEVICE:-cpu}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"
EVAL_BATCHES="${EVAL_BATCHES:-2}"

cmd=(
  "${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" matrix
  --device "${DEVICE}"
  --train-steps "${TRAIN_STEPS}"
  --eval-batches "${EVAL_BATCHES}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${ENWIK8_PATH}" ]]; then
  cmd+=(--enwik8-path "${ENWIK8_PATH}")
fi
"${cmd[@]}"
