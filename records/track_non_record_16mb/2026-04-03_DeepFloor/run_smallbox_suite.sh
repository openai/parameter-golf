#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/records/track_non_record_16mb/2026-04-03_DeepFloor/runs/smallbox}"
ENWIK8_PATH="${ENWIK8_PATH:-/workspace/data/enwik8}"
TRAIN_STEPS="${TRAIN_STEPS:-4}"
EVAL_BATCHES="${EVAL_BATCHES:-4}"

"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" \
  --python-bin "${PYTHON_BIN}" \
  local-unit
"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" \
  --python-bin "${PYTHON_BIN}" \
  local-smoke \
  --device cuda \
  --train-steps "${TRAIN_STEPS}" \
  --eval-batches "${EVAL_BATCHES}" \
  --enwik8-path "${ENWIK8_PATH}" \
  --output-dir "${OUTPUT_DIR}/smoke"
"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" \
  --python-bin "${PYTHON_BIN}" \
  matrix \
  --device cuda \
  --train-steps "${TRAIN_STEPS}" \
  --eval-batches "${EVAL_BATCHES}" \
  --enwik8-path "${ENWIK8_PATH}" \
  --output-dir "${OUTPUT_DIR}/matrix"
"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" \
  --python-bin "${PYTHON_BIN}" \
  report \
  --run-dir "${OUTPUT_DIR}/matrix"
