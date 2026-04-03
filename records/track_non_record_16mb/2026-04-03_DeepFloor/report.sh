#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_DIR="${RUN_DIR:-${ROOT}/records/track_non_record_16mb/2026-04-03_DeepFloor/runs/matrix}"

"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" \
  --python-bin "${PYTHON_BIN}" \
  report \
  --run-dir "${RUN_DIR}"
