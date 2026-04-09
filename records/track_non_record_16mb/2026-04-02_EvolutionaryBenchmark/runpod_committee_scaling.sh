#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

COMMITTEE_LOG="${EVO_COMMITTEE_LOG:-${EVO_OUTPUT_DIR}/committee_scaling_$(date -u +%Y%m%dT%H%M%SZ).log}"
EVO_COMMITTEE_EVAL_BATCHES="${EVO_COMMITTEE_EVAL_BATCHES:-8}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"

run_and_log() {
  echo "+ $*" | tee -a "${COMMITTEE_LOG}"
  "$@" 2>&1 | tee -a "${COMMITTEE_LOG}"
}

run_committee() {
  local copies="$1"
  local train_seconds="$2"
  local seed="$3"
  local topks="$4"
  local name="viability_member_ensemble_c${copies}_t${train_seconds}_bf16_seed${seed}.json"
  run_and_log \
    "${EVO_PYTHON_BIN}" \
    "${ROOT}/tools/evolutionary_benchmark.py" \
    crossover-viability \
    --device cuda \
    --dtype bf16 \
    --seed "${seed}" \
    --enwik8-path "${EVO_ENWIK8_PATH}" \
    --copies "${copies}" \
    --train-seconds "${train_seconds}" \
    --crossover-strategies layer_swap \
    --percentiles 50 \
    --eval-batches "${EVO_COMMITTEE_EVAL_BATCHES}" \
    --pair-limit 0 \
    --ensemble-topks "${topks}" \
    --member-train-mode parallel_vmap \
    --output-json "${EVO_OUTPUT_DIR}/${name}"
}

echo "Writing committee scaling log to ${COMMITTEE_LOG}"

run_committee 4 30 1337 2,4
run_committee 4 30 2025 2,4
run_committee 8 30 1337 2,4,8
run_committee 8 30 2025 2,4,8
run_committee 8 60 1337 2,4,8
run_committee 8 60 2025 2,4,8
run_committee 16 30 1337 2,4,8,16
run_committee 16 30 2025 2,4,8,16
run_committee 32 15 1337 2,4,8,16,32
run_committee 32 15 2025 2,4,8,16,32

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}"/viability_member_ensemble_c*_bf16_seed*.json \
  --section viability \
  --format table
