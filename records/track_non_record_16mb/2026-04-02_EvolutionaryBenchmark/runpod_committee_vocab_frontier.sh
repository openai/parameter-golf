#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

VOCAB_LOG="${EVO_VOCAB_FRONTIER_LOG:-${EVO_OUTPUT_DIR}/committee_vocab_frontier_$(date -u +%Y%m%dT%H%M%SZ).log}"
EVO_VOCAB_FRONTIER_EVAL_BATCHES="${EVO_VOCAB_FRONTIER_EVAL_BATCHES:-8}"
EVO_VOCAB_FRONTIER_SEEDS="${EVO_VOCAB_FRONTIER_SEEDS:-1337,2025}"
EVO_VOCAB_FRONTIER_STAGE_COPIES="${EVO_VOCAB_FRONTIER_STAGE_COPIES:-2,4,8}"
EVO_VOCAB_FRONTIER_STAGE_SECONDS="${EVO_VOCAB_FRONTIER_STAGE_SECONDS:-120,30,15}"
EVO_VOCAB_FRONTIER_TOPKS="${EVO_VOCAB_FRONTIER_TOPKS:-2,4,8}"
EVO_VOCAB_FRONTIER_TOKENIZER_NAME="${EVO_VOCAB_FRONTIER_TOKENIZER_NAME:-sp_bpe_1024}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"

run_and_log() {
  echo "+ $*" | tee -a "${VOCAB_LOG}"
  "$@" 2>&1 | tee -a "${VOCAB_LOG}"
}

have_run() {
  local path="$1"
  [[ -f "${path}" ]]
}

run_vocab_point() {
  local variant_name="$1"
  local seed="$2"
  shift 2
  local output_path="${EVO_OUTPUT_DIR}/committee_vocab_${variant_name}_bf16_seed${seed}.json"
  if have_run "${output_path}"; then
    echo "= skip $(basename "${output_path}")" | tee -a "${VOCAB_LOG}"
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
    --stage-copies "${EVO_VOCAB_FRONTIER_STAGE_COPIES}" \
    --stage-train-seconds "${EVO_VOCAB_FRONTIER_STAGE_SECONDS}" \
    --eval-batches "${EVO_VOCAB_FRONTIER_EVAL_BATCHES}" \
    --ensemble-topks "${EVO_VOCAB_FRONTIER_TOPKS}" \
    --spawn-noise-std 0.0 \
    --qk-gain-init 4.0 \
    --spine-variant xsa \
    --xsa-last-n 4 \
    "$@" \
    --output-json "${output_path}"
}

echo "Writing vocab frontier log to ${VOCAB_LOG}"

IFS=',' read -r -a seed_list <<< "${EVO_VOCAB_FRONTIER_SEEDS}"
for seed in "${seed_list[@]}"; do
  run_vocab_point bytes_xsa_qk4 "${seed}"
  run_vocab_point sp1024_xsa_qk4 "${seed}" \
    --tokenization-mode sentencepiece \
    --tokenizer-name "${EVO_VOCAB_FRONTIER_TOKENIZER_NAME}"
done

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}"/committee_vocab_*_bf16_seed*.json \
  --section viability \
  --format table
