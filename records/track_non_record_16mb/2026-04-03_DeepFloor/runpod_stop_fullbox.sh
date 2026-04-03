#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"

"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_runpod.py" stop "$@"
