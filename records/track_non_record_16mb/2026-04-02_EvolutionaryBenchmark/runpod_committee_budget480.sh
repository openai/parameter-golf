#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

BUDGET_LOG="${EVO_BUDGET480_LOG:-${EVO_OUTPUT_DIR}/committee_budget480_$(date -u +%Y%m%dT%H%M%SZ).log}"
EVO_BUDGET480_EVAL_BATCHES="${EVO_BUDGET480_EVAL_BATCHES:-8}"
EVO_BUDGET480_SEEDS="${EVO_BUDGET480_SEEDS:-1337,2025}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"

run_and_log() {
  echo "+ $*" | tee -a "${BUDGET_LOG}"
  "$@" 2>&1 | tee -a "${BUDGET_LOG}"
}

have_run() {
  local path="$1"
  [[ -f "${path}" ]]
}

run_budget_point() {
  local copies="$1"
  local train_seconds="$2"
  local seed="$3"
  local topks="$4"
  local name="viability_member_ensemble_c${copies}_t${train_seconds}_bf16_seed${seed}.json"
  local output_path="${EVO_OUTPUT_DIR}/${name}"
  if have_run "${output_path}"; then
    echo "= skip ${name}" | tee -a "${BUDGET_LOG}"
    return
  fi
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
    --eval-batches "${EVO_BUDGET480_EVAL_BATCHES}" \
    --pair-limit 0 \
    --ensemble-topks "${topks}" \
    --member-train-mode parallel_vmap \
    --output-json "${output_path}"
}

echo "Writing committee budget-480 log to ${BUDGET_LOG}"

IFS=',' read -r -a seed_list <<< "${EVO_BUDGET480_SEEDS}"
for seed in "${seed_list[@]}"; do
  run_budget_point 4 120 "${seed}" 2,4
  run_budget_point 6 80 "${seed}" 2,4,6
  run_budget_point 12 40 "${seed}" 2,4,8,12
  run_budget_point 24 20 "${seed}" 2,4,8,12,16,24
done

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}"/viability_member_ensemble_c*_bf16_seed*.json \
  --section viability \
  --format table
