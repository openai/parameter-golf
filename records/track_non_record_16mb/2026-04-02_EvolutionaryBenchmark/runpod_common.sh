#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../../.." && pwd)"

EVO_PYTHON_BIN="${EVO_PYTHON_BIN:-python3}"
EVO_ENWIK8_PATH="${EVO_ENWIK8_PATH:-/workspace/data/enwik8}"
EVO_OUTPUT_DIR="${EVO_OUTPUT_DIR:-${DIR}/runs}"
EVO_LOG_DIR="${EVO_LOG_DIR:-${EVO_OUTPUT_DIR}/logs}"
EVO_GPUS="${EVO_GPUS:-auto}"
EVO_MAX_WORKERS="${EVO_MAX_WORKERS:-0}"
EVO_SKIP_EXISTING="${EVO_SKIP_EXISTING:-1}"
EVO_FAIL_FAST="${EVO_FAIL_FAST:-0}"

evo_queue() {
  local extra_args=("$@")
  local cmd=(
    "${EVO_PYTHON_BIN}"
    "${ROOT}/tools/run_evolutionary_matrix.py"
    --python-bin "${EVO_PYTHON_BIN}"
    --script-path "${ROOT}/tools/evolutionary_benchmark.py"
    --output-dir "${EVO_OUTPUT_DIR}"
    --log-dir "${EVO_LOG_DIR}"
    --enwik8-path "${EVO_ENWIK8_PATH}"
    --gpus "${EVO_GPUS}"
    --max-workers "${EVO_MAX_WORKERS}"
  )
  if [[ "${EVO_SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi
  if [[ "${EVO_FAIL_FAST}" == "1" ]]; then
    cmd+=(--fail-fast)
  fi
  cmd+=("${extra_args[@]}")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

evo_summary() {
  local section="${1:-all}"
  "${EVO_PYTHON_BIN}" \
    "${ROOT}/tools/summarize_evolutionary_runs.py" \
    "${EVO_OUTPUT_DIR}" \
    --section "${section}"
}
