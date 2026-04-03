#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/records/track_non_record_16mb/2026-04-03_DeepFloor/runs/fullbox}"

"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" full-report --run-root "${RUN_ROOT}"
