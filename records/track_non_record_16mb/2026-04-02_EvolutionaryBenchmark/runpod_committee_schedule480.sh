#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

SCHEDULE_LOG="${EVO_SCHEDULE480_LOG:-${EVO_OUTPUT_DIR}/committee_schedule480_$(date -u +%Y%m%dT%H%M%SZ).log}"
EVO_SCHEDULE480_EVAL_BATCHES="${EVO_SCHEDULE480_EVAL_BATCHES:-8}"
EVO_SCHEDULE480_SEEDS="${EVO_SCHEDULE480_SEEDS:-1337,2025}"
EVO_SCHEDULE480_SPAWN_NOISE_STD="${EVO_SCHEDULE480_SPAWN_NOISE_STD:-0.0}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"

run_and_log() {
  echo "+ $*" | tee -a "${SCHEDULE_LOG}"
  "$@" 2>&1 | tee -a "${SCHEDULE_LOG}"
}

have_run() {
  local path="$1"
  [[ -f "${path}" ]]
}

run_schedule_point() {
  local schedule_name="$1"
  local stage_copies="$2"
  local stage_train_seconds="$3"
  local seed="$4"
  local topks="$5"
  local output_path="${EVO_OUTPUT_DIR}/committee_schedule_${schedule_name}_bf16_seed${seed}.json"
  if have_run "${output_path}"; then
    echo "= skip $(basename "${output_path}")" | tee -a "${SCHEDULE_LOG}"
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
    --eval-batches "${EVO_SCHEDULE480_EVAL_BATCHES}" \
    --ensemble-topks "${topks}" \
    --spawn-noise-std "${EVO_SCHEDULE480_SPAWN_NOISE_STD}" \
    --output-json "${output_path}"
}

echo "Writing committee schedule-480 log to ${SCHEDULE_LOG}"

IFS=',' read -r -a seed_list <<< "${EVO_SCHEDULE480_SEEDS}"
for seed in "${seed_list[@]}"; do
  run_schedule_point s4x90_8x15 4,8 90,15 "${seed}" 2,4,8
  run_schedule_point s4x60_8x30 4,8 60,30 "${seed}" 2,4,8
  run_schedule_point s4x30_8x45 4,8 30,45 "${seed}" 2,4,8
  run_schedule_point s4x60_16x15 4,16 60,15 "${seed}" 2,4,8,16
  run_schedule_point s2x120_4x30_8x15 2,4,8 120,30,15 "${seed}" 2,4,8
  run_schedule_point s4x60_8x15_16x7p5 4,8,16 60,15,7.5 "${seed}" 2,4,8,16
done

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}"/committee_schedule_*_bf16_seed*.json \
  --section viability \
  --format table
