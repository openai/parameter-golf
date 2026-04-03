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
SFW_NPROC_PER_NODE="${SFW_NPROC_PER_NODE:-8}"
SFW_LAUNCHER="${SFW_LAUNCHER:-auto}"
SFW_DEVICE="${SFW_DEVICE:-auto}"
SFW_OMP_NUM_THREADS="${SFW_OMP_NUM_THREADS:-1}"

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

sfw_run_profile() {
  local profile="$1"
  local seed="$2"
  shift 2

  local run_dir
  run_dir="$(sfw_make_run_dir "${profile}" "${seed}")"
  local log_path="${run_dir}/train.log"
  local result_path="${run_dir}/result.json"
  local seed_pool_path="${run_dir}/seed_pool.npz"
  local model_artifact_path="${run_dir}/model_int8.npz"
  local notes_path="${run_dir}/notes.txt"
  local command_path="${run_dir}/command.sh"

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
    --seed-pool-path "${seed_pool_path}"
    --model-artifact-path "${model_artifact_path}"
    --weight-decay "${SFW_WEIGHT_DECAY:-0.01}"
    --bank-store-mode "${SFW_BANK_STORE_MODE:-runtime}"
    --bank-runtime-dtype "${SFW_BANK_RUNTIME_DTYPE:-fp16}"
    --eval-append-writes-per-batch "${SFW_EVAL_APPEND_WRITES_PER_BATCH:-0}"
    --num-experts "${SFW_NUM_EXPERTS:-8}"
    --embed-dim "${SFW_EMBED_DIM:-128}"
    --expert-hidden "${SFW_EXPERT_HIDDEN:-128}"
    --expert-rank "${SFW_EXPERT_RANK:-16}"
    --fused-dim "${SFW_FUSED_DIM:-512}"
    --runtime-dim "${SFW_RUNTIME_DIM:-512}"
    --code-dim "${SFW_CODE_DIM:-64}"
    --query-dim "${SFW_QUERY_DIM:-128}"
    --reader-heads "${SFW_READER_HEADS:-8}"
    --topk "${SFW_TOPK:-64}"
  )
  if [[ "${SFW_EVAL_ONLINE_APPEND:-false}" == "true" ]]; then
    cmd+=(--eval-online-append)
  else
    cmd+=(--no-eval-online-append)
  fi
  if [[ -n "${SFW_SEED_POOL_LOAD_PATH:-}" ]]; then
    cmd+=(--seed-pool-load-path "${SFW_SEED_POOL_LOAD_PATH}")
  fi
  cmd+=("$@")

  {
    printf 'profile=%s\n' "${profile}"
    printf 'seed=%s\n' "${seed}"
    printf 'launcher=%s\n' "${SFW_LAUNCHER}"
    printf 'nproc_per_node=%s\n' "${SFW_NPROC_PER_NODE}"
    printf 'device=%s\n' "${SFW_DEVICE}"
    printf 'record_dir=%s\n' "${SFW_RECORD_DIR}"
    printf 'repo_root=%s\n' "${SFW_REPO_ROOT}"
  } > "${notes_path}"

  sfw_write_command_file "${command_path}" "${cmd[@]}"

  (
    cd "${SFW_RECORD_DIR}"
    export OMP_NUM_THREADS="${SFW_OMP_NUM_THREADS}"
    "${cmd[@]}"
  )

  printf 'Run completed: %s\n' "${run_dir}"
  printf '  log: %s\n' "${log_path}"
  printf '  result: %s\n' "${result_path}"
  printf '  seed pool: %s\n' "${seed_pool_path}"
  printf '  model artifact: %s\n' "${model_artifact_path}"
}
