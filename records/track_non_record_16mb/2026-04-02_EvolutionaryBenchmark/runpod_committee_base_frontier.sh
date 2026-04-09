#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

BASE_LOG="${EVO_BASE_FRONTIER_LOG:-${EVO_OUTPUT_DIR}/committee_base_frontier_$(date -u +%Y%m%dT%H%M%SZ).log}"
EVO_BASE_FRONTIER_EVAL_BATCHES="${EVO_BASE_FRONTIER_EVAL_BATCHES:-8}"
EVO_BASE_FRONTIER_SEEDS="${EVO_BASE_FRONTIER_SEEDS:-1337,2025}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"

run_and_log() {
  echo "+ $*" | tee -a "${BASE_LOG}"
  "$@" 2>&1 | tee -a "${BASE_LOG}"
}

have_run() {
  local path="$1"
  [[ -f "${path}" ]]
}

run_committee_variant() {
  local variant_name="$1"
  local seed="$2"
  shift 2
  local output_path="${EVO_OUTPUT_DIR}/committee_base_${variant_name}_c4_t120_bf16_seed${seed}.json"
  if have_run "${output_path}"; then
    echo "= skip $(basename "${output_path}")" | tee -a "${BASE_LOG}"
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
    --copies 4 \
    --train-seconds 120 \
    --crossover-strategies layer_swap \
    --percentiles 50 \
    --eval-batches "${EVO_BASE_FRONTIER_EVAL_BATCHES}" \
    --pair-limit 0 \
    --ensemble-topks 2,4 \
    --member-train-mode parallel_vmap \
    "$@" \
    --output-json "${output_path}"
}

echo "Writing committee base-frontier log to ${BASE_LOG}"

IFS=',' read -r -a seed_list <<< "${EVO_BASE_FRONTIER_SEEDS}"
for seed in "${seed_list[@]}"; do
  run_committee_variant plain_qk4 "${seed}" \
    --qk-gain-init 4.0
  run_committee_variant xsa_qk1p5 "${seed}" \
    --spine-variant xsa \
    --xsa-last-n 4
  run_committee_variant xsa_qk4 "${seed}" \
    --qk-gain-init 4.0 \
    --spine-variant xsa \
    --xsa-last-n 4
done

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}"/committee_base_*_c4_t120_bf16_seed*.json \
  --section viability \
  --format table
