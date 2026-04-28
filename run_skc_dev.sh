#!/usr/bin/env bash
# Research / SKC-full dev launcher.
# - No COMPETITION_PROFILE, no RUNTIME_PATH_POLICY=strict: MoE, feedback,
#   capsules, Koopman speculator, adaptive halt stay live.
# - STAGE = precompile | train | both  (default: both)
# - On compile-stage OOM, retry stage A once with COMPILE_MODE=reduce-overhead
#   and INDUCTOR_DISABLE_CONSTANT_FOLDING=1.
# - Optional upload-before-launch via POD_SSH + LOCAL_WORKTREE.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
mkdir -p logs/skc_dev

STAGE="${STAGE:-both}"
case "${STAGE}" in
  precompile|train|both) ;;
  *) echo "STAGE must be one of: precompile, train, both (got '${STAGE}')." >&2; exit 2 ;;
esac

RUN_ID="${RUN_ID:-skc_dev_$(date +%Y%m%d_%H%M%S)}"
NPROC="${NPROC:-2}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot auto-size VRAM." >&2
  exit 2
fi

sleep 2

SMI_ARGS=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  SMI_ARGS="-i ${CUDA_VISIBLE_DEVICES}"
fi

FREE_MB_MIN="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits $SMI_ARGS | head -n "${NPROC}" | awk 'NR==1{m=$1} $1<m{m=$1} END{print int(m)}')"
if [[ -z "${FREE_MB_MIN}" || "${FREE_MB_MIN}" -le 0 ]]; then
  echo "Unable to read free GPU memory." >&2
  exit 3
fi

pick_train_batch_tokens() {
  local free_mb="$1"
  if (( free_mb >= 70000 )); then echo 196608
  elif (( free_mb >= 50000 )); then echo 131072
  elif (( free_mb >= 35000 )); then echo 98304
  elif (( free_mb >= 30000 )); then echo 65536
  elif (( free_mb >= 24000 )); then echo 49152
  else echo 32768
  fi
}

pick_warmup_batch_tokens() {
  local free_mb="$1"
  if (( free_mb < 30000 )); then echo 8192
  elif (( free_mb < 50000 )); then echo 16384
  else echo 0
  fi
}

TRAIN_BATCH_TOKENS_AUTO="$(pick_train_batch_tokens "${FREE_MB_MIN}")"
COMPILER_WARMUP_BATCH_TOKENS_AUTO="$(pick_warmup_batch_tokens "${FREE_MB_MIN}")"

LOG_FILE="logs/skc_dev/${RUN_ID}.log"
PRECOMPILE_LOG="logs/skc_dev/${RUN_ID}.precompile.log"
TRAIN_LOG="logs/skc_dev/${RUN_ID}.train.log"

# Optional: upload canonical sources to a remote pod before launch.
LOCAL_WORKTREE="${LOCAL_WORKTREE:-}"
if [[ -n "${LOCAL_WORKTREE}" && -n "${POD_SSH:-}" ]]; then
  SSH_KEY="${POD_SSH_KEY:-${HOME}/.ssh/id_ed25519_runpod}"
  echo "uploading canonical sources to root@${POD_SSH}:/workspace/" | tee "${LOG_FILE}"
  scp -i "${SSH_KEY}" -P "${POD_SSH_PORT:-22}" \
    "${LOCAL_WORKTREE}/train_gpt_verbose.py" \
    "${LOCAL_WORKTREE}/triton_kernels.py" \
    "${LOCAL_WORKTREE}/build_submission.py" \
    "root@${POD_SSH}:/workspace/" | tee -a "${LOG_FILE}"
fi

# Common dev env. Deliberately omits COMPETITION_PROFILE and
# RUNTIME_PATH_POLICY=strict so the full research path runs.
COMMON_ENV=(
  RUN_ID="${RUN_ID}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"
  OMP_NUM_THREADS=8
  DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp8192}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_8192_bpe.model}"
  HARD_BUDGET_BYTES="${HARD_BUDGET_BYTES:-16000000}"
  HARD_BUDGET_ENFORCE="${HARD_BUDGET_ENFORCE:-0}"
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-570}"
  ITERATIONS="${ITERATIONS:-200000}"
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-${TRAIN_BATCH_TOKENS_AUTO}}"
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
  SLIDING_EVAL="${SLIDING_EVAL:-1}"
  SLIDING_BATCH_SIZE="${SLIDING_BATCH_SIZE:-64}"
  FINAL_EVAL_SEQUENTIAL_CARRY="${FINAL_EVAL_SEQUENTIAL_CARRY:-1}"
  TORCH_NCCL_TIMEOUT_SEC=3600
  COMPILE_MODE="${COMPILE_MODE:-max-autotune}"
  COMPILER_WARMUP_STEPS="${COMPILER_WARMUP_STEPS:-1}"
  SYNTHETIC_WARMUP="${SYNTHETIC_WARMUP:-1}"
  COMPILER_WARMUP_BATCH_TOKENS="${COMPILER_WARMUP_BATCH_TOKENS:-${COMPILER_WARMUP_BATCH_TOKENS_AUTO}}"
  INDUCTOR_DISABLE_CONSTANT_FOLDING="${INDUCTOR_DISABLE_CONSTANT_FOLDING:-0}"
  TORCHINDUCTOR_FX_GRAPH_CACHE=1
  TORCHINDUCTOR_AUTOGRAD_CACHE=1
  COMPILE_SHAPE_PADDING="${COMPILE_SHAPE_PADDING:-1}"
  COMPILE_TRITON_CUDAGRAPHS="${COMPILE_TRITON_CUDAGRAPHS:-0}"
  DIAGNOSTICS_ENABLED="${DIAGNOSTICS_ENABLED:-0}"
  WALL_CLOCK_TIMEOUT="${WALL_CLOCK_TIMEOUT:-570}"
)

echo "run_id=${RUN_ID} stage=${STAGE} nproc=${NPROC} free_mb_min=${FREE_MB_MIN} train_batch_tokens=${TRAIN_BATCH_TOKENS_AUTO} warmup_batch_tokens=${COMPILER_WARMUP_BATCH_TOKENS_AUTO}" | tee -a "${LOG_FILE}"

run_precompile() {
  local mode="$1"
  local disable_fold="$2"
  local logf="$3"
  local stage_env=(
    "${COMMON_ENV[@]}"
    PRECOMPILE_ONLY=1
    COMPILE_MODE="${mode}"
    INDUCTOR_DISABLE_CONSTANT_FOLDING="${disable_fold}"
  )
  echo "precompile: mode=${mode} disable_constant_folding=${disable_fold}" | tee -a "${logf}"
  timeout 900 env "${stage_env[@]}" torchrun --standalone --nproc_per_node="${NPROC}" train_gpt_verbose.py 2>&1 | tee -a "${logf}"
  return "${PIPESTATUS[0]}"
}

run_train() {
  local logf="$1"
  local stage_env=(
    "${COMMON_ENV[@]}"
    PRECOMPILE_ONLY=0
  )
  echo "train: mode=${COMPILE_MODE:-max-autotune}" | tee -a "${logf}"
  timeout 1800 env "${stage_env[@]}" torchrun --standalone --nproc_per_node="${NPROC}" train_gpt_verbose.py 2>&1 | tee -a "${logf}"
}

if [[ "${STAGE}" == "precompile" || "${STAGE}" == "both" ]]; then
  : > "${PRECOMPILE_LOG}"
  set +e
  run_precompile "${COMPILE_MODE:-max-autotune}" "${INDUCTOR_DISABLE_CONSTANT_FOLDING:-0}" "${PRECOMPILE_LOG}"
  PC_RC=$?
  set -e
  if (( PC_RC != 0 )) && grep -q "OutOfMemoryError" "${PRECOMPILE_LOG}"; then
    echo "precompile OOM detected; retrying with reduce-overhead + constant-folding off" | tee -a "${LOG_FILE}"
    run_precompile "reduce-overhead" "1" "${PRECOMPILE_LOG}"
    PC_RC=$?
    # Propagate reduced-overhead choice into the train stage.
    export COMPILE_MODE=reduce-overhead
    export INDUCTOR_DISABLE_CONSTANT_FOLDING=1
    # Rebuild COMMON_ENV to reflect the fallback for stage B.
    COMMON_ENV=(
      "${COMMON_ENV[@]//COMPILE_MODE=*/COMPILE_MODE=reduce-overhead}"
    )
  fi
  if (( PC_RC != 0 )); then
    echo "precompile failed rc=${PC_RC}; see ${PRECOMPILE_LOG}" >&2
    exit "${PC_RC}"
  fi
fi

if [[ "${STAGE}" == "train" || "${STAGE}" == "both" ]]; then
  : > "${TRAIN_LOG}"
  run_train "${TRAIN_LOG}"
fi

echo "Done. Logs: ${LOG_FILE} ${PRECOMPILE_LOG} ${TRAIN_LOG}"
