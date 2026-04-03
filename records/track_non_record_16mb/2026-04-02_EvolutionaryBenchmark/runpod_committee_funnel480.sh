#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

FUNNEL_LOG="${EVO_FUNNEL480_LOG:-${EVO_OUTPUT_DIR}/committee_funnel480_$(date -u +%Y%m%dT%H%M%SZ).log}"
EVO_FUNNEL480_EVAL_BATCHES="${EVO_FUNNEL480_EVAL_BATCHES:-8}"
EVO_FUNNEL480_SEEDS="${EVO_FUNNEL480_SEEDS:-1337,2025}"
EVO_FUNNEL480_SPAWN_NOISE_STD="${EVO_FUNNEL480_SPAWN_NOISE_STD:-0.0}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"

run_and_log() {
  echo "+ $*" | tee -a "${FUNNEL_LOG}"
  "$@" 2>&1 | tee -a "${FUNNEL_LOG}"
}

have_run() {
  local path="$1"
  [[ -f "${path}" ]]
}

run_funnel_point() {
  local schedule_name="$1"
  local stage_copies="$2"
  local stage_train_seconds="$3"
  local seed="$4"
  local topks="$5"
  local output_path="${EVO_OUTPUT_DIR}/committee_funnel_${schedule_name}_bf16_seed${seed}.json"
  if have_run "${output_path}"; then
    echo "= skip $(basename "${output_path}")" | tee -a "${FUNNEL_LOG}"
    return
  fi
  run_and_log \
    "${EVO_PYTHON_BIN}" \
    "${ROOT}/tools/evolutionary_benchmark.py" \
    committee-schedule \
    --device cuda \
    --dtype bf16 \
    --seed "${seed}" \
    --enwik8-path "${EVO_ENWIK8_PATH}" \
    --stage-copies "${stage_copies}" \
    --stage-train-seconds "${stage_train_seconds}" \
    --eval-batches "${EVO_FUNNEL480_EVAL_BATCHES}" \
    --ensemble-topks "${topks}" \
    --spawn-noise-std "${EVO_FUNNEL480_SPAWN_NOISE_STD}" \
    --output-json "${output_path}"
}

echo "Writing committee funnel-480 log to ${FUNNEL_LOG}"

IFS=',' read -r -a seed_list <<< "${EVO_FUNNEL480_SEEDS}"
for seed in "${seed_list[@]}"; do
  run_funnel_point f2x90_4x30_8x15_4x15 2,4,8,4 90,30,15,15 "${seed}" 2,4
  run_funnel_point f2x60_4x30_8x15_4x30 2,4,8,4 60,30,15,30 "${seed}" 2,4
  run_funnel_point f4x60_8x15_4x30 4,8,4 60,15,30 "${seed}" 2,4
  run_funnel_point f4x45_8x15_4x30_2x30 4,8,4,2 45,15,30,30 "${seed}" 2
done

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}"/committee_funnel_*_bf16_seed*.json \
  --section viability \
  --format table
