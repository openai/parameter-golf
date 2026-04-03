#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
LEASE_MINUTES="${LEASE_MINUTES:-60}"
ARGS=("$@")
HAS_LEASE=0

for arg in "${ARGS[@]}"; do
  if [[ "${arg}" == "--lease-minutes" ]]; then
    HAS_LEASE=1
    break
  fi
done

if [[ "${HAS_LEASE}" -eq 0 ]]; then
  ARGS=(--lease-minutes "${LEASE_MINUTES}" "${ARGS[@]}")
fi

"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_runpod.py" extend "${ARGS[@]}"
