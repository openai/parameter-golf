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
SFW_NPROC_PER_NODE="${SFW_NPROC_PER_NODE:-auto}"
SFW_LAUNCHER="${SFW_LAUNCHER:-auto}"
SFW_DEVICE="${SFW_DEVICE:-auto}"
SFW_OMP_NUM_THREADS="${SFW_OMP_NUM_THREADS:-1}"
SFW_TARGET_HARDWARE="${SFW_TARGET_HARDWARE:-auto}"
SFW_TARGET_GPU_COUNT="${SFW_TARGET_GPU_COUNT:-auto}"

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

sfw_resolve_target_gpu_count() {
  local detected_gpu_count="$1"
  if [[ "${SFW_TARGET_GPU_COUNT}" == "auto" ]]; then
    printf '%s\n' "${detected_gpu_count}"
    return
  fi
  printf '%s\n' "${SFW_TARGET_GPU_COUNT}"
}

sfw_resolve_target_hardware() {
  local detected_gpu_count="$1"
  local target_gpu_count="$2"
  if [[ "${SFW_TARGET_HARDWARE}" != "auto" ]]; then
    printf '%s\n' "${SFW_TARGET_HARDWARE}"
    return
  fi
  if (( detected_gpu_count > 0 )); then
    printf '%sxGPU auto-detect\n' "${target_gpu_count}"
  else
    printf 'cpu auto-detect\n'
  fi
}

sfw_resolve_nproc_per_node() {
  local detected_gpu_count="$1"
  local target_gpu_count="$2"
  if [[ "${SFW_NPROC_PER_NODE}" != "auto" ]]; then
    printf '%s\n' "${SFW_NPROC_PER_NODE}"
    return
  fi
  if (( detected_gpu_count <= 1 || target_gpu_count <= 1 )); then
    printf '1\n'
    return
  fi
  if (( detected_gpu_count < target_gpu_count )); then
    printf '%s\n' "${detected_gpu_count}"
  else
    printf '%s\n' "${target_gpu_count}"
  fi
}

sfw_apply_bool_flag() {
  local flag_true="$1"
  local flag_false="$2"
  local value="$3"
  local -n out_ref="$4"
  case "${value}" in
    1|true|TRUE|yes|YES|on|ON)
      out_ref+=("${flag_true}")
      ;;
    0|false|FALSE|no|NO|off|OFF)
      out_ref+=("${flag_false}")
      ;;
  esac
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
  local notes_path="${run_dir}/notes.txt"
  local command_path="${run_dir}/command.sh"
  local detected_gpu_count
  detected_gpu_count="$(sfw_detect_gpu_count)"
  local target_gpu_count
  target_gpu_count="$(sfw_resolve_target_gpu_count "${detected_gpu_count}")"
  local target_hardware
  target_hardware="$(sfw_resolve_target_hardware "${detected_gpu_count}" "${target_gpu_count}")"
  local resolved_nproc_per_node
  resolved_nproc_per_node="$(sfw_resolve_nproc_per_node "${detected_gpu_count}" "${target_gpu_count}")"

  local -a cmd
  if [[ "${SFW_LAUNCHER}" == "python" || ( "${SFW_LAUNCHER}" == "auto" && "${resolved_nproc_per_node}" == "1" ) ]]; then
    cmd=("${SFW_PYTHON_BIN}" "${SFW_RECORD_DIR}/train_gpt.py")
  else
    if ! command -v torchrun >/dev/null 2>&1; then
      echo "[error] torchrun is required when launcher=${SFW_LAUNCHER} and nproc_per_node=${resolved_nproc_per_node}" >&2
      return 1
    fi
    cmd=(torchrun --standalone --nproc_per_node "${resolved_nproc_per_node}" "${SFW_RECORD_DIR}/train_gpt.py")
  fi

  cmd+=(
    --device "${SFW_DEVICE}"
    --seed "${seed}"
    --auto-log-path "${log_path}"
    --output-json "${result_path}"
    --model-artifact-path "${model_artifact_path}"
    --weight-decay "${SFW_WEIGHT_DECAY:-0.04}"
    --train-tokens "${SFW_TRAIN_TOKENS:-262144}"
    --val-tokens "${SFW_VAL_TOKENS:-65536}"
    --train-steps "${SFW_TRAIN_STEPS:-256}"
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
    --spine-variant "${SFW_SPINE_VARIANT:-xsa}"
    --xsa-last-n "${SFW_XSA_LAST_N:-4}"
    --base-lr "${SFW_BASE_LR:-0.002}"
    --warmup-steps "${SFW_WARMUP_STEPS:-30}"
    --memory-orders "${SFW_MEMORY_ORDERS:-1,2,3,4}"
    --memory-table-size "${SFW_MEMORY_TABLE_SIZE:-65536}"
    --memory-update-lr "${SFW_MEMORY_UPDATE_LR:-0.05}"
    --memory-decay "${SFW_MEMORY_DECAY:-0.999}"
    --memory-ema-decay "${SFW_MEMORY_EMA_DECAY:-0.95}"
    --memory-read-scale "${SFW_MEMORY_READ_SCALE:-1.0}"
    --memory-min-read-count "${SFW_MEMORY_MIN_READ_COUNT:-2}"
    --memory-max-delta-norm "${SFW_MEMORY_MAX_DELTA_NORM:-4.0}"
    --maintenance-passes "${SFW_MAINTENANCE_PASSES:-1}"
    --maintenance-mode "${SFW_MAINTENANCE_MODE:-replay}"
    --maintenance-blend "${SFW_MAINTENANCE_BLEND:-0.25}"
    --maintenance-step-size "${SFW_MAINTENANCE_STEP_SIZE:-0.05}"
    --maintenance-max-slots "${SFW_MAINTENANCE_MAX_SLOTS:-64}"
    --maintenance-metric "${SFW_MAINTENANCE_METRIC:-loss}"
    --maintenance-grad-mix "${SFW_MAINTENANCE_GRAD_MIX:-0.25}"
    --maintenance-loss-ema-decay "${SFW_MAINTENANCE_LOSS_EMA_DECAY:-0.95}"
    --maintenance-replay-depth "${SFW_MAINTENANCE_REPLAY_DEPTH:-2}"
    --maintenance-replay-candidates "${SFW_MAINTENANCE_REPLAY_CANDIDATES:-64}"
  )

  if [[ -n "${SFW_MEMORY_ORDER_SCALES:-}" ]]; then
    cmd+=(--memory-order-scales "${SFW_MEMORY_ORDER_SCALES}")
  fi

  sfw_apply_bool_flag "--maintenance-use-grad" "--no-maintenance-use-grad" "${SFW_MAINTENANCE_USE_GRAD:-0}" cmd

  cmd+=("$@")

  {
    printf 'profile=%s\n' "${profile}"
    printf 'seed=%s\n' "${seed}"
    printf 'launcher=%s\n' "${SFW_LAUNCHER}"
    printf 'requested_nproc_per_node=%s\n' "${SFW_NPROC_PER_NODE}"
    printf 'resolved_nproc_per_node=%s\n' "${resolved_nproc_per_node}"
    printf 'target_hardware=%s\n' "${target_hardware}"
    printf 'requested_target_gpu_count=%s\n' "${SFW_TARGET_GPU_COUNT}"
    printf 'resolved_target_gpu_count=%s\n' "${target_gpu_count}"
    printf 'detected_gpu_count=%s\n' "${detected_gpu_count}"
    printf 'device=%s\n' "${SFW_DEVICE}"
    printf 'record_dir=%s\n' "${SFW_RECORD_DIR}"
    printf 'repo_root=%s\n' "${SFW_REPO_ROOT}"
  } > "${notes_path}"

  sfw_write_command_file "${command_path}" "${cmd[@]}"

  if [[ "${detected_gpu_count}" != "${target_gpu_count}" ]]; then
    echo "[warn] target hardware is '${target_hardware}' (${target_gpu_count} GPU(s)) but detected ${detected_gpu_count} GPU(s)." | tee -a "${notes_path}"
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
}
