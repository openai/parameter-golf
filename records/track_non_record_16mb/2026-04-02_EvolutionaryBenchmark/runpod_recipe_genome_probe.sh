#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

RECIPE_LOG="${EVO_RECIPE_LOG:-${EVO_OUTPUT_DIR}/recipe_genome_probe_$(date -u +%Y%m%dT%H%M%SZ).log}"
EVO_RECIPE_SEEDS="${EVO_RECIPE_SEEDS:-1337,2025}"
EVO_RECIPE_EVAL_BATCHES="${EVO_RECIPE_EVAL_BATCHES:-8}"
EVO_RECIPE_POPULATION="${EVO_RECIPE_POPULATION:-12}"
EVO_RECIPE_GENERATIONS="${EVO_RECIPE_GENERATIONS:-6}"
EVO_RECIPE_TOURNAMENT="${EVO_RECIPE_TOURNAMENT:-3}"
EVO_RECIPE_TRAIN_SECONDS="${EVO_RECIPE_TRAIN_SECONDS:-90}"
EVO_RECIPE_CONFIRM_TOPK="${EVO_RECIPE_CONFIRM_TOPK:-3}"
EVO_RECIPE_CONFIRM_SECONDS="${EVO_RECIPE_CONFIRM_SECONDS:-180}"
EVO_RECIPE_MUTATION_RATE="${EVO_RECIPE_MUTATION_RATE:-0.2}"
EVO_RECIPE_ARTIFACT_LIMIT_MB="${EVO_RECIPE_ARTIFACT_LIMIT_MB:-16.0}"
EVO_RECIPE_PROFILE="${EVO_RECIPE_PROFILE:-frontier}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"
require_benchmark_python_modules
require_sentencepiece_tokenizer "sp_bpe_1024"

run_and_log() {
  echo "+ $*" | tee -a "${RECIPE_LOG}"
  "$@" 2>&1 | tee -a "${RECIPE_LOG}"
}

have_run() {
  local path="$1"
  [[ -f "${path}" ]]
}

run_recipe_seed() {
  local seed="$1"
  local output_path="${EVO_OUTPUT_DIR}/recipe_evolution_${EVO_RECIPE_PROFILE}_bf16_seed${seed}.json"
  if have_run "${output_path}"; then
    echo "= skip $(basename "${output_path}")" | tee -a "${RECIPE_LOG}"
    return
  fi
  run_and_log \
    "${EVO_PYTHON_BIN}" \
    "${ROOT}/tools/evolutionary_benchmark.py" \
    recipe-evolution \
    --device cuda \
    --dtype bf16 \
    --seed "${seed}" \
    --enwik8-path "${EVO_ENWIK8_PATH}" \
    --population-size "${EVO_RECIPE_POPULATION}" \
    --generations "${EVO_RECIPE_GENERATIONS}" \
    --tournament-size "${EVO_RECIPE_TOURNAMENT}" \
    --train-seconds "${EVO_RECIPE_TRAIN_SECONDS}" \
    --eval-batches "${EVO_RECIPE_EVAL_BATCHES}" \
    --mutation-rate "${EVO_RECIPE_MUTATION_RATE}" \
    --artifact-limit-mb "${EVO_RECIPE_ARTIFACT_LIMIT_MB}" \
    --recipe-profile "${EVO_RECIPE_PROFILE}" \
    --confirm-topk "${EVO_RECIPE_CONFIRM_TOPK}" \
    --confirm-train-seconds "${EVO_RECIPE_CONFIRM_SECONDS}" \
    --output-json "${output_path}"
}

echo "Writing recipe-genome probe log to ${RECIPE_LOG}"

IFS=',' read -r -a seed_list <<< "${EVO_RECIPE_SEEDS}"
for seed in "${seed_list[@]}"; do
  run_recipe_seed "${seed}"
done

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}"/recipe_evolution_*_bf16_seed*.json \
  --section evolution \
  --format table
