#!/usr/bin/env bash
set -euo pipefail

SFW_RECORD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFW_REPO_ROOT="$(cd "${SFW_RECORD_DIR}/../../.." && pwd)"
SFW_RUNS_DIR="${SFW_RUNS_DIR:-${SFW_RECORD_DIR}/runs}"
if [[ -x "${SFW_REPO_ROOT}/.venv/bin/python" ]]; then
  SFW_DEFAULT_PYTHON_BIN="${SFW_REPO_ROOT}/.venv/bin/python"
else
  SFW_DEFAULT_PYTHON_BIN="python3"
fi
SFW_PYTHON_BIN="${SFW_PYTHON_BIN:-${SFW_DEFAULT_PYTHON_BIN}}"
SFW_NPROC_PER_NODE="${SFW_NPROC_PER_NODE:-1}"
SFW_LAUNCHER="${SFW_LAUNCHER:-auto}"
SFW_DEVICE="${SFW_DEVICE:-auto}"
SFW_OMP_NUM_THREADS="${SFW_OMP_NUM_THREADS:-1}"
SFW_TARGET_HARDWARE="${SFW_TARGET_HARDWARE:-1xH100 exploratory}"
SFW_TARGET_GPU_COUNT="${SFW_TARGET_GPU_COUNT:-1}"

sfw_timestamp() {
  date -u +"%Y%m%dT%H%M%SZ"
}

sfw_run_name() {
  local profile="$1"
  local seed="$2"
  local stamp="${SFW_TIMESTAMP:-$(sfw_timestamp)}"
  local suffix="${SFW_RUN_SUFFIX:-}"
  local name="${stamp}_${profile}_seed${seed}"
  if [[ -n "${suffix}" ]]; then
    name="${name}_${suffix}"
  fi
  printf '%s\n' "${name}"
}

sfw_make_run_dir() {
  local profile="$1"
  local seed="$2"
  local run_dir="${SFW_RUNS_DIR}/$(sfw_run_name "${profile}" "${seed}")"
  mkdir -p "${run_dir}"
  printf '%s\n' "${run_dir}"
}

sfw_write_command_file() {
  local command_path="$1"
  shift
  {
    printf 'cd %q\n' "${SFW_RECORD_DIR}"
    printf 'OMP_NUM_THREADS=%q ' "${SFW_OMP_NUM_THREADS}"
    printf '%q ' "$@"
    printf '\n'
  } > "${command_path}"
  chmod +x "${command_path}"
}

sfw_detect_gpu_count() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '0\n'
    return
  fi
  local count
  count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
  if [[ -z "${count}" ]]; then
    count="0"
  fi
  printf '%s\n' "${count}"
}

sfw_run_profile() {
  local profile="$1"
  local seed="$2"
  shift 2

  local run_dir
  run_dir="$(sfw_make_run_dir "${profile}" "${seed}")"
  local log_path="${run_dir}/train.log"
  local result_path="${run_dir}/result.json"
  local model_artifact_path="${run_dir}/model_int8.npz"
  local residual_artifact_path="${run_dir}/residual_tables.npz"
  local notes_path="${run_dir}/notes.txt"
  local command_path="${run_dir}/command.sh"
  local detected_gpu_count
  detected_gpu_count="$(sfw_detect_gpu_count)"

  local -a cmd
  if [[ "${SFW_LAUNCHER}" == "python" || ( "${SFW_LAUNCHER}" == "auto" && "${SFW_NPROC_PER_NODE}" == "1" ) ]]; then
    cmd=("${SFW_PYTHON_BIN}" "${SFW_RECORD_DIR}/train_gpt.py")
  else
    cmd=(torchrun --standalone --nproc_per_node "${SFW_NPROC_PER_NODE}" "${SFW_RECORD_DIR}/train_gpt.py")
  fi

  cmd+=(
    --device "${SFW_DEVICE}"
    --seed "${seed}"
    --auto-log-path "${log_path}"
    --output-json "${result_path}"
    --model-artifact-path "${model_artifact_path}"
    --residual-artifact-path "${residual_artifact_path}"
    --weight-decay "${SFW_WEIGHT_DECAY:-0.04}"
    --train-tokens "${SFW_TRAIN_TOKENS:-262144}"
    --val-tokens "${SFW_VAL_TOKENS:-65536}"
    --train-steps "${SFW_TRAIN_STEPS:-256}"
    --calibration-steps "${SFW_CALIBRATION_STEPS:-128}"
    --eval-batches "${SFW_EVAL_BATCHES:-128}"
    --batch-size "${SFW_BATCH_SIZE:-8}"
    --seq-len "${SFW_SEQ_LEN:-512}"
    --stride "${SFW_STRIDE:-64}"
    --report-every "${SFW_REPORT_EVERY:-8}"
    --model-dim "${SFW_MODEL_DIM:-512}"
    --num-layers "${SFW_NUM_LAYERS:-11}"
    --num-heads "${SFW_NUM_HEADS:-8}"
    --num-kv-heads "${SFW_NUM_KV_HEADS:-4}"
    --mlp-mult "${SFW_MLP_MULT:-3}"
    --spine-variant "${SFW_SPINE_VARIANT:-plain}"
    --xsa-last-n "${SFW_XSA_LAST_N:-0}"
    --residual-rank "${SFW_RESIDUAL_RANK:-16}"
    --residual-orders "${SFW_RESIDUAL_ORDERS:-1,2,3,4}"
    --residual-table-size "${SFW_RESIDUAL_TABLE_SIZE:-65536}"
    --residual-update-lr "${SFW_RESIDUAL_UPDATE_LR:-0.05}"
    --residual-decay "${SFW_RESIDUAL_DECAY:-0.999}"
    --base-lr "${SFW_BASE_LR:-0.002}"
    --basis-lr "${SFW_BASIS_LR:-0.01}"
    --warmup-steps "${SFW_WARMUP_STEPS:-30}"
  )
  cmd+=("$@")

  {
    printf 'profile=%s\n' "${profile}"
    printf 'seed=%s\n' "${seed}"
    printf 'launcher=%s\n' "${SFW_LAUNCHER}"
    printf 'nproc_per_node=%s\n' "${SFW_NPROC_PER_NODE}"
    printf 'target_hardware=%s\n' "${SFW_TARGET_HARDWARE}"
    printf 'target_gpu_count=%s\n' "${SFW_TARGET_GPU_COUNT}"
    printf 'detected_gpu_count=%s\n' "${detected_gpu_count}"
    printf 'device=%s\n' "${SFW_DEVICE}"
    printf 'record_dir=%s\n' "${SFW_RECORD_DIR}"
    printf 'repo_root=%s\n' "${SFW_REPO_ROOT}"
  } > "${notes_path}"

  sfw_write_command_file "${command_path}" "${cmd[@]}"

  if [[ "${detected_gpu_count}" != "${SFW_TARGET_GPU_COUNT}" ]]; then
    echo "[warn] target hardware is '${SFW_TARGET_HARDWARE}' (${SFW_TARGET_GPU_COUNT} GPU(s)) but detected ${detected_gpu_count} GPU(s)." | tee -a "${notes_path}"
    echo "[warn] override SFW_TARGET_GPU_COUNT / SFW_TARGET_HARDWARE / SFW_NPROC_PER_NODE if this is intentional." | tee -a "${notes_path}"
  fi

  (
    cd "${SFW_RECORD_DIR}"
    export OMP_NUM_THREADS="${SFW_OMP_NUM_THREADS}"
    "${cmd[@]}"
  )

  printf 'Run completed: %s\n' "${run_dir}"
  printf '  log: %s\n' "${log_path}"
  printf '  result: %s\n' "${result_path}"
  printf '  model artifact: %s\n' "${model_artifact_path}"
  printf '  residual artifact: %s\n' "${residual_artifact_path}"
}
