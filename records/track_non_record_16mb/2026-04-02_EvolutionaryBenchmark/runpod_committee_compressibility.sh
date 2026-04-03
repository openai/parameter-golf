#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

COMPRESS_LOG="${EVO_COMPRESS_LOG:-${EVO_OUTPUT_DIR}/committee_compressibility_$(date -u +%Y%m%dT%H%M%SZ).log}"
EVO_COMPRESS_SEEDS="${EVO_COMPRESS_SEEDS:-1337,2025}"
EVO_COMPRESS_STAGE_COPIES="${EVO_COMPRESS_STAGE_COPIES:-2,4,8}"
EVO_COMPRESS_STAGE_SECONDS="${EVO_COMPRESS_STAGE_SECONDS:-120,30,15}"
EVO_COMPRESS_TOPKS="${EVO_COMPRESS_TOPKS:-2,4,8}"
EVO_COMPRESS_EVAL_BATCHES="${EVO_COMPRESS_EVAL_BATCHES:-8}"
EVO_COMPRESS_ARTIFACT_LIMIT_MB="${EVO_COMPRESS_ARTIFACT_LIMIT_MB:-16.0}"
EVO_COMPRESS_DELTA_BUDGET_FRACTIONS="${EVO_COMPRESS_DELTA_BUDGET_FRACTIONS:-0.25,0.5,1.0}"
EVO_COMPRESS_BASIS_RANKS="${EVO_COMPRESS_BASIS_RANKS:-1,2,4}"
EVO_COMPRESS_ANALYSIS_TOPK="${EVO_COMPRESS_ANALYSIS_TOPK:-8192}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"

run_and_log() {
  echo "+ $*" | tee -a "${COMPRESS_LOG}"
  "$@" 2>&1 | tee -a "${COMPRESS_LOG}"
}

have_run() {
  local path="$1"
  [[ -f "${path}" ]]
}

run_compress_seed() {
  local seed="$1"
  local output_path="${EVO_OUTPUT_DIR}/committee_compressibility_xsa_qk4_bf16_seed${seed}.json"
  if have_run "${output_path}"; then
    echo "= skip $(basename "${output_path}")" | tee -a "${COMPRESS_LOG}"
    return
  fi
  run_and_log \
    "${EVO_PYTHON_BIN}" \
    "${ROOT}/tools/evolutionary_benchmark.py" \
    committee-compressibility \
    --device cuda \
    --dtype bf16 \
    --seed "${seed}" \
    --enwik8-path "${EVO_ENWIK8_PATH}" \
    --stage-copies "${EVO_COMPRESS_STAGE_COPIES}" \
    --stage-train-seconds "${EVO_COMPRESS_STAGE_SECONDS}" \
    --ensemble-topks "${EVO_COMPRESS_TOPKS}" \
    --eval-batches "${EVO_COMPRESS_EVAL_BATCHES}" \
    --spawn-noise-std 0.0 \
    --artifact-limit-mb "${EVO_COMPRESS_ARTIFACT_LIMIT_MB}" \
    --delta-budget-fractions "${EVO_COMPRESS_DELTA_BUDGET_FRACTIONS}" \
    --basis-ranks "${EVO_COMPRESS_BASIS_RANKS}" \
    --analysis-topk "${EVO_COMPRESS_ANALYSIS_TOPK}" \
    --qk-gain-init 4.0 \
    --spine-variant xsa \
    --xsa-last-n 4 \
    --output-json "${output_path}"
}

echo "Writing committee compressibility log to ${COMPRESS_LOG}"

IFS=',' read -r -a seed_list <<< "${EVO_COMPRESS_SEEDS}"
for seed in "${seed_list[@]}"; do
  run_compress_seed "${seed}"
done

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}"/committee_compressibility_*_bf16_seed*.json \
  --section viability \
  --format table
