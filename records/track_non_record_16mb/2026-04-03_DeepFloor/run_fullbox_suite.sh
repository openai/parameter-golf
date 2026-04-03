#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/records/track_non_record_16mb/2026-04-03_DeepFloor/runs/fullbox}"
ENWIK8_PATH="${ENWIK8_PATH:-/workspace/data/enwik8}"
FULLBOX_GPUS="${FULLBOX_GPUS:-0,1,2,3,4,5,6,7}"
SMOKE_GPU="${SMOKE_GPU:-0}"
MATRIX_GPU="${MATRIX_GPU:-1}"
SMOKE_TRAIN_STEPS="${SMOKE_TRAIN_STEPS:-4}"
SMOKE_EVAL_BATCHES="${SMOKE_EVAL_BATCHES:-4}"
MATRIX_TRAIN_STEPS="${MATRIX_TRAIN_STEPS:-4}"
MATRIX_EVAL_BATCHES="${MATRIX_EVAL_BATCHES:-4}"
EVO_PROFILES="${EVO_PROFILES:-compact,frontier}"
EVO_SEEDS="${EVO_SEEDS:-1337,2025,4242,5151}"
EVO_POPULATION="${EVO_POPULATION:-12}"
EVO_GENERATIONS="${EVO_GENERATIONS:-6}"
EVO_TOURNAMENT="${EVO_TOURNAMENT:-3}"
EVO_TRAIN_STEPS="${EVO_TRAIN_STEPS:-16}"
EVO_EVAL_BATCHES="${EVO_EVAL_BATCHES:-8}"
EVO_MUTATION_RATE="${EVO_MUTATION_RATE:-0.2}"
EVO_ARTIFACT_LIMIT_MB="${EVO_ARTIFACT_LIMIT_MB:-16.0}"
EVO_CONFIRM_TOPK="${EVO_CONFIRM_TOPK:-3}"
EVO_CONFIRM_TRAIN_STEPS="${EVO_CONFIRM_TRAIN_STEPS:-32}"
EVO_SKIP_EXISTING="${EVO_SKIP_EXISTING:-1}"
LAUNCH_LOG_DIR="${OUTPUT_DIR}/launch_logs"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/smoke" "${OUTPUT_DIR}/matrix" "${OUTPUT_DIR}/evolution" "${LAUNCH_LOG_DIR}"

echo "[deepfloor] output_dir=${OUTPUT_DIR}"
echo "[deepfloor] enwik8_path=${ENWIK8_PATH}"
echo "[deepfloor] gpus=${FULLBOX_GPUS}"

"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" \
  --python-bin "${PYTHON_BIN}" \
  local-unit

echo "[gate] smoke on gpu=${SMOKE_GPU}"
CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" \
  "${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" \
  --python-bin "${PYTHON_BIN}" \
  local-smoke \
  --device cuda \
  --train-steps "${SMOKE_TRAIN_STEPS}" \
  --eval-batches "${SMOKE_EVAL_BATCHES}" \
  --enwik8-path "${ENWIK8_PATH}" \
  --output-dir "${OUTPUT_DIR}/smoke"

echo "[gate] matrix on gpu=${MATRIX_GPU}"
CUDA_VISIBLE_DEVICES="${MATRIX_GPU}" \
  "${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" \
  --python-bin "${PYTHON_BIN}" \
  matrix \
  --device cuda \
  --train-steps "${MATRIX_TRAIN_STEPS}" \
  --eval-batches "${MATRIX_EVAL_BATCHES}" \
  --enwik8-path "${ENWIK8_PATH}" \
  --output-dir "${OUTPUT_DIR}/matrix"

IFS=',' read -r -a gpu_list <<< "${FULLBOX_GPUS}"
IFS=',' read -r -a profile_list <<< "${EVO_PROFILES}"
IFS=',' read -r -a seed_list <<< "${EVO_SEEDS}"

job_labels=()
job_profiles=()
job_seeds=()
for profile in "${profile_list[@]}"; do
  for seed in "${seed_list[@]}"; do
    job_labels+=("deepfloor_recipe_${profile}_seed${seed}")
    job_profiles+=("${profile}")
    job_seeds+=("${seed}")
  done
done

if [[ "${#job_labels[@]}" -eq 0 ]]; then
  echo "[error] no evolution jobs requested" >&2
  exit 1
fi

batch_width="${#gpu_list[@]}"
if [[ "${batch_width}" -le 0 ]]; then
  echo "[error] FULLBOX_GPUS must list at least one GPU id" >&2
  exit 1
fi

launch_batch() {
  local start_idx="$1"
  local -a pids=()
  local -a labels=()
  local slot=0
  while [[ "${slot}" -lt "${batch_width}" ]]; do
    local job_idx=$((start_idx + slot))
    if [[ "${job_idx}" -ge "${#job_labels[@]}" ]]; then
      break
    fi
    local gpu="${gpu_list[$slot]}"
    local label="${job_labels[$job_idx]}"
    local profile="${job_profiles[$job_idx]}"
    local seed="${job_seeds[$job_idx]}"
    local output_json="${OUTPUT_DIR}/evolution/${label}.json"
    local log_path="${LAUNCH_LOG_DIR}/${label}.log"
    if [[ "${EVO_SKIP_EXISTING}" == "1" && -f "${output_json}" ]]; then
      echo "= skip ${label}"
      slot=$((slot + 1))
      continue
    fi
    echo "[launch] gpu=${gpu} profile=${profile} seed=${seed} label=${label}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" \
      --python-bin "${PYTHON_BIN}" \
      evolution \
      --device cuda \
      --seed "${seed}" \
      --population-size "${EVO_POPULATION}" \
      --generations "${EVO_GENERATIONS}" \
      --tournament-size "${EVO_TOURNAMENT}" \
      --train-steps "${EVO_TRAIN_STEPS}" \
      --eval-batches "${EVO_EVAL_BATCHES}" \
      --mutation-rate "${EVO_MUTATION_RATE}" \
      --artifact-limit-mb "${EVO_ARTIFACT_LIMIT_MB}" \
      --deepfloor-profile "${profile}" \
      --confirm-topk "${EVO_CONFIRM_TOPK}" \
      --confirm-train-steps "${EVO_CONFIRM_TRAIN_STEPS}" \
      --enwik8-path "${ENWIK8_PATH}" \
      --output-json "${output_json}" \
      > "${log_path}" 2>&1 &
    pids+=("$!")
    labels+=("${label}")
    slot=$((slot + 1))
  done

  local batch_status=0
  for idx in "${!pids[@]}"; do
    local pid="${pids[$idx]}"
    local label="${labels[$idx]}"
    if ! wait "${pid}"; then
      echo "[error] evolution job failed: ${label} (pid=${pid})" >&2
      batch_status=1
    fi
  done
  return "${batch_status}"
}

for ((start_idx = 0; start_idx < ${#job_labels[@]}; start_idx += batch_width)); do
  launch_batch "${start_idx}"
done

"${PYTHON_BIN}" "${ROOT}/tools/run_deepfloor_suite.py" \
  --python-bin "${PYTHON_BIN}" \
  full-report \
  --run-root "${OUTPUT_DIR}"
