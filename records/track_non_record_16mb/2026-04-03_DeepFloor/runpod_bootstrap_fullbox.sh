#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
SCRIPT_PATH="${ROOT}/records/track_non_record_16mb/2026-04-03_DeepFloor/run_fullbox_suite.sh"

"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_runpod.py" bootstrap --script-path "${SCRIPT_PATH}" "$@"
