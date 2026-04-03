#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

PARALLEL_LOG="${EVO_BASE_PARALLEL_LOG:-${EVO_OUTPUT_DIR}/committee_base_parallel_$(date -u +%Y%m%dT%H%M%SZ).log}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"

run_job() {
  local gpu="$1"
  local name="$2"
  shift 2
  echo "[launch] gpu=${gpu} name=${name}" | tee -a "${PARALLEL_LOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${EVO_PYTHON_BIN}" \
    "${ROOT}/tools/evolutionary_benchmark.py" \
    crossover-viability \
    --device cuda \
    --dtype bf16 \
    --enwik8-path "${EVO_ENWIK8_PATH}" \
    --copies 4 \
    --train-seconds 120 \
    --crossover-strategies layer_swap \
    --percentiles 50 \
    --eval-batches 8 \
    --pair-limit 0 \
    --ensemble-topks 2,4 \
    --member-train-mode parallel_vmap \
    "$@" \
    --output-json "${EVO_OUTPUT_DIR}/${name}.json" \
    > "${EVO_LOG_DIR}/${name}.log" 2>&1 &
}

launch_spec() {
  local gpu="$1"
  local name="$2"
  shift 2
  run_job "${gpu}" "${name}" "$@"
}

launch_spec 0 committee_base_plain_qk4_c4_t120_bf16_seed1337 \
  --seed 1337 --qk-gain-init 4.0
launch_spec 1 committee_base_xsa_qk1p5_c4_t120_bf16_seed1337 \
  --seed 1337 --spine-variant xsa --xsa-last-n 4
launch_spec 2 committee_base_xsa_qk4_c4_t120_bf16_seed1337 \
  --seed 1337 --qk-gain-init 4.0 --spine-variant xsa --xsa-last-n 4
launch_spec 3 committee_base_plain_qk4_c4_t120_bf16_seed2025 \
  --seed 2025 --qk-gain-init 4.0
launch_spec 4 committee_base_xsa_qk1p5_c4_t120_bf16_seed2025 \
  --seed 2025 --spine-variant xsa --xsa-last-n 4

wait -n

launch_spec 0 committee_base_xsa_qk4_c4_t120_bf16_seed2025 \
  --seed 2025 --qk-gain-init 4.0 --spine-variant xsa --xsa-last-n 4

wait

"${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}"/committee_base_*_c4_t120_bf16_seed*.json \
  --section viability \
  --format table | tee -a "${PARALLEL_LOG}"
