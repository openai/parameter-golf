#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

FOLLOWUP_LOG="${EVO_FOLLOWUP_LOG:-${EVO_OUTPUT_DIR}/focused_followup_$(date -u +%Y%m%dT%H%M%SZ).log}"
EVO_PROBE_BASE_TRAIN_SECONDS="${EVO_PROBE_BASE_TRAIN_SECONDS:-30}"
EVO_PROBE_GENERATIONS="${EVO_PROBE_GENERATIONS:-6}"
EVO_PROBE_POPULATION="${EVO_PROBE_POPULATION:-16}"
EVO_PROBE_TOURNAMENT="${EVO_PROBE_TOURNAMENT:-4}"
EVO_PROBE_EVAL_BATCHES="${EVO_PROBE_EVAL_BATCHES:-8}"
EVO_PROBE_MUTATION_STD="${EVO_PROBE_MUTATION_STD:-5e-4}"
EVO_PROBE_MUTATION_FRACTION="${EVO_PROBE_MUTATION_FRACTION:-0.05}"
EVO_PROBE_ENSEMBLE_TOPKS="${EVO_PROBE_ENSEMBLE_TOPKS:-4,8,16}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"

run_and_log() {
  echo "+ $*" | tee -a "${FOLLOWUP_LOG}"
  "$@" 2>&1 | tee -a "${FOLLOWUP_LOG}"
}

echo "Writing focused follow-up log to ${FOLLOWUP_LOG}"

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/evolutionary_benchmark.py" \
  vmap-throughput \
  --device cuda \
  --dtype bf16 \
  --population-scales 8,64,256,1024,4096,16384 \
  --population-chunk-size 256 \
  --output-json "${EVO_OUTPUT_DIR}/throughput_bf16_standard_chunk256.json"

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/evolutionary_benchmark.py" \
  vmap-throughput \
  --device cuda \
  --dtype bf16 \
  --population-scales 256,1024,4096,16384 \
  --population-chunk-size 128 \
  --output-json "${EVO_OUTPUT_DIR}/throughput_bf16_standard_chunk128.json"

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/evolutionary_benchmark.py" \
  vmap-throughput \
  --device cuda \
  --dtype bf16 \
  --population-scales 256,1024,4096,16384 \
  --population-chunk-size 512 \
  --output-json "${EVO_OUTPUT_DIR}/throughput_bf16_standard_chunk512.json"

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/evolutionary_benchmark.py" \
  evolution-loop \
  --device cuda \
  --dtype bf16 \
  --enwik8-path "${EVO_ENWIK8_PATH}" \
  --base-train-seconds "${EVO_PROBE_BASE_TRAIN_SECONDS}" \
  --generations "${EVO_PROBE_GENERATIONS}" \
  --population-size "${EVO_PROBE_POPULATION}" \
  --tournament-size "${EVO_PROBE_TOURNAMENT}" \
  --crossover-strategy parent_copy \
  --mutation-std "${EVO_PROBE_MUTATION_STD}" \
  --mutation-fraction "${EVO_PROBE_MUTATION_FRACTION}" \
  --eval-batches "${EVO_PROBE_EVAL_BATCHES}" \
  --ensemble-topks "${EVO_PROBE_ENSEMBLE_TOPKS}" \
  --output-json "${EVO_OUTPUT_DIR}/evo_parent_copy_probe_bf16_seed1337.json"

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/evolutionary_benchmark.py" \
  evolution-loop \
  --device cuda \
  --dtype bf16 \
  --seed 2025 \
  --enwik8-path "${EVO_ENWIK8_PATH}" \
  --base-train-seconds "${EVO_PROBE_BASE_TRAIN_SECONDS}" \
  --generations "${EVO_PROBE_GENERATIONS}" \
  --population-size "${EVO_PROBE_POPULATION}" \
  --tournament-size "${EVO_PROBE_TOURNAMENT}" \
  --crossover-strategy parent_copy \
  --mutation-std "${EVO_PROBE_MUTATION_STD}" \
  --mutation-fraction "${EVO_PROBE_MUTATION_FRACTION}" \
  --eval-batches "${EVO_PROBE_EVAL_BATCHES}" \
  --ensemble-topks "${EVO_PROBE_ENSEMBLE_TOPKS}" \
  --output-json "${EVO_OUTPUT_DIR}/evo_parent_copy_probe_bf16_seed2025.json"

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/evolutionary_benchmark.py" \
  evolution-loop \
  --device cuda \
  --dtype bf16 \
  --enwik8-path "${EVO_ENWIK8_PATH}" \
  --base-train-seconds "${EVO_PROBE_BASE_TRAIN_SECONDS}" \
  --generations "${EVO_PROBE_GENERATIONS}" \
  --population-size "${EVO_PROBE_POPULATION}" \
  --tournament-size "${EVO_PROBE_TOURNAMENT}" \
  --crossover-strategy delta_overlap \
  --crossover-percentile 50 \
  --mutation-std "${EVO_PROBE_MUTATION_STD}" \
  --mutation-fraction "${EVO_PROBE_MUTATION_FRACTION}" \
  --eval-batches "${EVO_PROBE_EVAL_BATCHES}" \
  --ensemble-topks "${EVO_PROBE_ENSEMBLE_TOPKS}" \
  --output-json "${EVO_OUTPUT_DIR}/evo_delta_overlap_probe_bf16_seed1337.json"

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}/throughput_bf16_standard_chunk256.json" \
  "${EVO_OUTPUT_DIR}/throughput_bf16_standard_chunk128.json" \
  "${EVO_OUTPUT_DIR}/throughput_bf16_standard_chunk512.json" \
  "${EVO_OUTPUT_DIR}/evo_parent_copy_probe_bf16_seed1337.json" \
  "${EVO_OUTPUT_DIR}/evo_parent_copy_probe_bf16_seed2025.json" \
  "${EVO_OUTPUT_DIR}/evo_delta_overlap_probe_bf16_seed1337.json" \
  --format table
