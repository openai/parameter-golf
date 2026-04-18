#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
mkdir -p logs/max_vram_10min

RUN_ID="${RUN_ID:-max_vram_10min_$(date +%Y%m%d_%H%M%S)}"
NPROC="${NPROC:-2}"
AUTO_TUNE="${AUTO_TUNE:-0}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-2100}"
AUTO_TUNE_PROFILE="${AUTO_TUNE_PROFILE:-}"
if [[ -z "${AUTO_TUNE_PROFILE}" ]]; then
  if [[ "${DIAGNOSTICS_ENABLED:-0}" == "1" ]]; then
    AUTO_TUNE_PROFILE="diagnostic"
  else
    AUTO_TUNE_PROFILE="competition"
  fi
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot auto-size VRAM." >&2
  exit 2
fi

# Allow any zombie processes from previous failures to clear VRAM.
sleep 2
nvidia-smi --gpu-reset 2>/dev/null || true

SMI_ARGS=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  SMI_ARGS="-i ${CUDA_VISIBLE_DEVICES}"
fi

FREE_MB_EXPECTED_MIN="${FREE_MB_EXPECTED_MIN:-70000}"
for _ in 1 2 3; do
  sleep 3
  FREE_CHECK="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits $SMI_ARGS | head -n "${NPROC}" | awk 'NR==1{m=$1} $1<m{m=$1} END{print int(m)}')"
  if [[ -n "${FREE_CHECK}" ]] && (( FREE_CHECK >= FREE_MB_EXPECTED_MIN )); then
    break
  fi
done

# Use the minimum free VRAM across the GPUs we will use.
FREE_MB_MIN="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits $SMI_ARGS | head -n "${NPROC}" | awk 'NR==1{m=$1} $1<m{m=$1} END{print int(m)}')"
if [[ -z "${FREE_MB_MIN}" || "${FREE_MB_MIN}" -le 0 ]]; then
  echo "Unable to read free GPU memory." >&2
  exit 3
fi

pick_train_batch_tokens() {
  local free_mb="$1"
  # 24 GB cards cannot hold full-size compile buffers alongside the model.
  # Leave compile headroom on sub-30GB devices.
  if (( free_mb >= 70000 )); then echo 196608
  elif (( free_mb >= 50000 )); then echo 131072
  elif (( free_mb >= 35000 )); then echo 98304
  elif (( free_mb >= 30000 )); then echo 65536
  elif (( free_mb >= 24000 )); then echo 49152
  elif (( free_mb >= 16000 )); then echo 32768
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
if [[ -n "${MATRIX_LOCK_BATCH_TOKENS:-}" ]]; then
  TRAIN_BATCH_TOKENS_AUTO="${MATRIX_LOCK_BATCH_TOKENS}"
fi
COMPILER_WARMUP_BATCH_TOKENS_AUTO="$(pick_warmup_batch_tokens "${FREE_MB_MIN}")"
# Keep eval deterministic and OOM-safe for submission pipeline.
if (( FREE_MB_MIN >= 50000 )); then
  SLIDING_BATCH_SIZE_AUTO=128
else
  SLIDING_BATCH_SIZE_AUTO=64
fi

LOG_FILE="logs/max_vram_10min/${RUN_ID}.log"
AUTO_TUNE_ENV_FILE="logs/max_vram_10min/${RUN_ID}.auto_tune.env"
AUTO_TUNE_JSON_FILE="logs/max_vram_10min/${RUN_ID}.auto_tune.json"

echo "run_id=${RUN_ID} nproc=${NPROC} free_mb_min=${FREE_MB_MIN} train_batch_tokens=${TRAIN_BATCH_TOKENS_AUTO} warmup_batch_tokens=${COMPILER_WARMUP_BATCH_TOKENS_AUTO} sliding_batch=${SLIDING_BATCH_SIZE_AUTO}" | tee "${LOG_FILE}"

if [[ "${AUTO_TUNE}" == "1" ]]; then
  python3 auto_tune_launcher.py \
    --root "${ROOT_DIR}" \
    --run-id "${RUN_ID}" \
    --nproc "${NPROC}" \
    --profile "${AUTO_TUNE_PROFILE}" \
    --logs-dir "logs/auto_tune" \
    --emit-env-file "${AUTO_TUNE_ENV_FILE}" \
    --emit-json-file "${AUTO_TUNE_JSON_FILE}" | tee -a "${LOG_FILE}"
  # Auto-tune is restricted to runtime launcher knobs. It must not alter model shape
  # or artifact-budget parameters that determine the 16MB submission target.
  # shellcheck disable=SC1090
  source "${AUTO_TUNE_ENV_FILE}"
fi

COMMON_ENV=(
  RUN_ID="${RUN_ID}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-0}"
  OMP_NUM_THREADS=8
  DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp8192}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_8192_bpe.model}"
  COMPETITION_PROFILE="${COMPETITION_PROFILE:-1}"
  EXPORT_MODE="${EXPORT_MODE:-competition_gptq}"
  RUNTIME_PATH_POLICY="${RUNTIME_PATH_POLICY:-strict}"
  HARD_BUDGET_BYTES="${HARD_BUDGET_BYTES:-16000000}"
  HARD_BUDGET_ENFORCE="${HARD_BUDGET_ENFORCE:-1}"
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-570}"
  ITERATIONS="${ITERATIONS:-200000}"
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-${TRAIN_BATCH_TOKENS_AUTO}}"
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
  SLIDING_EVAL="${SLIDING_EVAL:-1}"
  SLIDING_BATCH_SIZE="${SLIDING_BATCH_SIZE:-${SLIDING_BATCH_SIZE_AUTO}}"
  FINAL_EVAL_SEQUENTIAL_CARRY="${FINAL_EVAL_SEQUENTIAL_CARRY:-1}"
  TORCH_NCCL_TIMEOUT_SEC="${TORCH_NCCL_TIMEOUT_SEC:-3600}"
  COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
  COMPILER_WARMUP_STEPS="${COMPILER_WARMUP_STEPS:-1}"
  SYNTHETIC_WARMUP="${SYNTHETIC_WARMUP:-1}"
  COMPILER_WARMUP_BATCH_TOKENS="${COMPILER_WARMUP_BATCH_TOKENS:-${COMPILER_WARMUP_BATCH_TOKENS_AUTO}}"
  INDUCTOR_DISABLE_CONSTANT_FOLDING="${INDUCTOR_DISABLE_CONSTANT_FOLDING:-0}"
  DIAGNOSTICS_ENABLED="${DIAGNOSTICS_ENABLED:-0}"
  TORCHINDUCTOR_FX_GRAPH_CACHE=1
  TORCHINDUCTOR_AUTOGRAD_CACHE=1
  COMPILE_SHAPE_PADDING=1
  COMPILE_TRITON_CUDAGRAPHS=1
  TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torch
  TRITON_CACHE_DIR=/workspace/cache/triton
  TERNARY_THRESHOLD_SEARCH=1
  TERNARY_SCALE_SEARCH=1
  EXPORT_ALIGNED_TRAIN="${EXPORT_ALIGNED_TRAIN:-1}"
  EXPORT_ALIGNED_TRAIN_START_FRACTION="${EXPORT_ALIGNED_TRAIN_START_FRACTION:-0.75}"
  EXPORT_PROXY_EVAL="${EXPORT_PROXY_EVAL:-1}"
  EXPORT_PROXY_EVERY="${EXPORT_PROXY_EVERY:-250}"
  EXPORT_PROXY_NUM_SEQS="${EXPORT_PROXY_NUM_SEQS:-16}"
  TERNARY_COMPRESS_BROTLI="${TERNARY_COMPRESS_BROTLI:-1}"
  ENGRAM_COMPETITION_ENABLED="${ENGRAM_COMPETITION_ENABLED:-0}"
  BIGRAM_HASH_ENABLED="${BIGRAM_HASH_ENABLED:-${ENGRAM_COMPETITION_ENABLED}}"
  SKC_RECURRENT_CORE="${SKC_RECURRENT_CORE:-1}"
  SKC_RESIDUAL_SCALE_INIT="${SKC_RESIDUAL_SCALE_INIT:-0.15}"
  SKC_AMP_RAMP_FRACTION="${SKC_AMP_RAMP_FRACTION:-0.3}"
  SKC_STRUCT_LR_MULT="${SKC_STRUCT_LR_MULT:-1.5}"
  HEAD_LR_MULT="${HEAD_LR_MULT:-1.0}"
  ENGRAM_TAPER_START="${ENGRAM_TAPER_START:-0.9}"
  ENGRAM_TAPER_END="${ENGRAM_TAPER_END:-0.99}"
  ENG_WRITE_EVERY="${ENG_WRITE_EVERY:-1}"
  ENGRAM_EXPORT_PRUNE_ENABLED="${ENGRAM_EXPORT_PRUNE_ENABLED:-1}"
  ENGRAM_EXPORT_KEEP_BIGRAM_RATIO="${ENGRAM_EXPORT_KEEP_BIGRAM_RATIO:-0.45}"
  ENGRAM_EXPORT_KEEP_TRIGRAM_RATIO="${ENGRAM_EXPORT_KEEP_TRIGRAM_RATIO:-0.20}"
  ENGRAM_EXPORT_KEEP_MIN_BUCKETS="${ENGRAM_EXPORT_KEEP_MIN_BUCKETS:-256}"
  ENGRAM_EXPORT_SCORE_ALPHA="${ENGRAM_EXPORT_SCORE_ALPHA:-0.80}"
  ENGRAM_EXPORT_TOKEN_BUDGET="${ENGRAM_EXPORT_TOKEN_BUDGET:-131072}"
  WALL_CLOCK_TIMEOUT="${MAX_WALLCLOCK_SECONDS:-570}"
)

# Optional: upload canonical sources to a remote pod before launch.
# Invoke by setting POD_SSH, POD_SSH_PORT (default 22), LOCAL_WORKTREE.
LOCAL_WORKTREE="${LOCAL_WORKTREE:-}"
if [[ -n "${LOCAL_WORKTREE}" && -n "${POD_SSH:-}" ]]; then
  SSH_KEY="${POD_SSH_KEY:-${HOME}/.ssh/id_ed25519_runpod}"
  scp -i "${SSH_KEY}" -P "${POD_SSH_PORT:-22}" \
    "${LOCAL_WORKTREE}/train_gpt_verbose.py" \
    "${LOCAL_WORKTREE}/triton_kernels.py" \
    "${LOCAL_WORKTREE}/build_submission.py" \
    "root@${POD_SSH}:/workspace/" | tee -a "${LOG_FILE}"
fi

if [[ "${SKIP_BUILD_SUBMISSION:-0}" != "1" ]]; then
  env "${COMMON_ENV[@]}" python3 build_submission.py >/dev/null
fi

timeout "${RUN_TIMEOUT_SECONDS}" env "${COMMON_ENV[@]}" torchrun --standalone --nproc_per_node="${NPROC}" train_gpt_verbose.py 2>&1 | tee -a "${LOG_FILE}"

echo "Done. Log: ${LOG_FILE}"
