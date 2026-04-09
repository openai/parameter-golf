#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
POD_ID="${1:?usage: $0 <pod-id>}"
shift || true

REMOTE_PATH="${REMOTE_PATH:-/workspace/parameter-golf/records/track_non_record_16mb/2026-04-03_DeepFloor/runs/smallbox/}"
LOCAL_PATH="${LOCAL_PATH:-${ROOT}/records/track_non_record_16mb/2026-04-03_DeepFloor/runs/smallbox/}"

"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_runpod.py" sync-path \
  "${POD_ID}" \
  --remote-path "${REMOTE_PATH}" \
  --local-path "${LOCAL_PATH}" \
  "$@"
