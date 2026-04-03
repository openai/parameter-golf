#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

ADAPTIVE_LOG="${EVO_ADAPTIVE_FRONTIER_LOG:-${EVO_OUTPUT_DIR}/committee_adaptive_frontier_$(date -u +%Y%m%dT%H%M%SZ).log}"
EVO_ADAPTIVE_FRONTIER_EVAL_BATCHES="${EVO_ADAPTIVE_FRONTIER_EVAL_BATCHES:-8}"
EVO_ADAPTIVE_FRONTIER_SEEDS="${EVO_ADAPTIVE_FRONTIER_SEEDS:-1337,2025}"
EVO_ADAPTIVE_FRONTIER_ROUND_SECONDS="${EVO_ADAPTIVE_FRONTIER_ROUND_SECONDS:-120,30,15,15}"
EVO_ADAPTIVE_FRONTIER_ENSEMBLE_TOPKS="${EVO_ADAPTIVE_FRONTIER_ENSEMBLE_TOPKS:-2,4,8,16}"
EVO_ADAPTIVE_FRONTIER_MIN_COPIES="${EVO_ADAPTIVE_FRONTIER_MIN_COPIES:-2}"
EVO_ADAPTIVE_FRONTIER_MAX_COPIES="${EVO_ADAPTIVE_FRONTIER_MAX_COPIES:-16}"
EVO_ADAPTIVE_FRONTIER_WIDEN_FACTOR="${EVO_ADAPTIVE_FRONTIER_WIDEN_FACTOR:-2}"
EVO_ADAPTIVE_FRONTIER_NARROW_FACTOR="${EVO_ADAPTIVE_FRONTIER_NARROW_FACTOR:-2}"
EVO_ADAPTIVE_FRONTIER_DEPTH_GAIN="${EVO_ADAPTIVE_FRONTIER_DEPTH_GAIN:-0.01}"
EVO_ADAPTIVE_FRONTIER_BREADTH_GAIN="${EVO_ADAPTIVE_FRONTIER_BREADTH_GAIN:-0.02}"
EVO_ADAPTIVE_FRONTIER_ARCHIVE_HIT_RATE="${EVO_ADAPTIVE_FRONTIER_ARCHIVE_HIT_RATE:-0.25}"
EVO_ADAPTIVE_FRONTIER_WINNER_CONC="${EVO_ADAPTIVE_FRONTIER_WINNER_CONC:-0.85}"
EVO_ADAPTIVE_FRONTIER_REPLAY_DISAGREE="${EVO_ADAPTIVE_FRONTIER_REPLAY_DISAGREE:-0.02}"
EVO_ADAPTIVE_FRONTIER_PAIRWISE_DISTANCE="${EVO_ADAPTIVE_FRONTIER_PAIRWISE_DISTANCE:-0.01}"

mkdir -p "${EVO_OUTPUT_DIR}" "${EVO_LOG_DIR}"

run_and_log() {
  echo "+ $*" | tee -a "${ADAPTIVE_LOG}"
  "$@" 2>&1 | tee -a "${ADAPTIVE_LOG}"
}

have_run() {
  local path="$1"
  [[ -f "${path}" ]]
}

run_adaptive_point() {
  local variant_name="$1"
  local seed="$2"
  shift 2
  local output_path="${EVO_OUTPUT_DIR}/committee_adaptive_${variant_name}_bf16_seed${seed}.json"
  if have_run "${output_path}"; then
    echo "= skip $(basename "${output_path}")" | tee -a "${ADAPTIVE_LOG}"
    return
  fi
  run_and_log \
    "${EVO_PYTHON_BIN}" \
    "${ROOT}/tools/evolutionary_benchmark.py" \
    committee-adaptive \
    --device cuda \
    --dtype bf16 \
    --seed "${seed}" \
    --enwik8-path "${EVO_ENWIK8_PATH}" \
    --initial-copies "${EVO_ADAPTIVE_FRONTIER_MIN_COPIES}" \
    --round-train-seconds "${EVO_ADAPTIVE_FRONTIER_ROUND_SECONDS}" \
    --eval-batches "${EVO_ADAPTIVE_FRONTIER_EVAL_BATCHES}" \
    --ensemble-topks "${EVO_ADAPTIVE_FRONTIER_ENSEMBLE_TOPKS}" \
    --spawn-noise-std 0.0 \
    --min-copies "${EVO_ADAPTIVE_FRONTIER_MIN_COPIES}" \
    --max-copies "${EVO_ADAPTIVE_FRONTIER_MAX_COPIES}" \
    --widen-factor "${EVO_ADAPTIVE_FRONTIER_WIDEN_FACTOR}" \
    --narrow-factor "${EVO_ADAPTIVE_FRONTIER_NARROW_FACTOR}" \
    --depth-gain-threshold "${EVO_ADAPTIVE_FRONTIER_DEPTH_GAIN}" \
    --breadth-gain-threshold "${EVO_ADAPTIVE_FRONTIER_BREADTH_GAIN}" \
    --archive-hit-rate-threshold "${EVO_ADAPTIVE_FRONTIER_ARCHIVE_HIT_RATE}" \
    --winner-concentration-threshold "${EVO_ADAPTIVE_FRONTIER_WINNER_CONC}" \
    --replay-disagreement-threshold "${EVO_ADAPTIVE_FRONTIER_REPLAY_DISAGREE}" \
    --pairwise-distance-threshold "${EVO_ADAPTIVE_FRONTIER_PAIRWISE_DISTANCE}" \
    "$@" \
    --output-json "${output_path}"
}

echo "Writing adaptive frontier log to ${ADAPTIVE_LOG}"

IFS=',' read -r -a seed_list <<< "${EVO_ADAPTIVE_FRONTIER_SEEDS}"
for seed in "${seed_list[@]}"; do
  run_adaptive_point xsa_qk4 "${seed}" \
    --qk-gain-init 4.0 \
    --spine-variant xsa \
    --xsa-last-n 4
done

run_and_log \
  "${EVO_PYTHON_BIN}" \
  "${ROOT}/tools/summarize_evolutionary_runs.py" \
  "${EVO_OUTPUT_DIR}"/committee_adaptive_*_bf16_seed*.json \
  --section viability \
  --format table
