#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
mkdir -p logs/max_vram_10min

RUN_ID="${RUN_ID:-max_vram_10min_$(date +%Y%m%d_%H%M%S)}"
NPROC="${NPROC:-2}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot auto-size VRAM." >&2
  exit 2
fi

# Use the minimum free VRAM across the GPUs we will use.
FREE_MB_MIN="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n "${NPROC}" | awk 'NR==1{m=$1} $1<m{m=$1} END{print int(m)}')"
if [[ -z "${FREE_MB_MIN}" || "${FREE_MB_MIN}" -le 0 ]]; then
  echo "Unable to read free GPU memory." >&2
  exit 3
fi

pick_train_batch_tokens() {
  local free_mb="$1"
  if (( free_mb >= 70000 )); then echo 196608
  elif (( free_mb >= 50000 )); then echo 131072
  elif (( free_mb >= 35000 )); then echo 98304
  elif (( free_mb >= 24000 )); then echo 65536
  elif (( free_mb >= 16000 )); then echo 49152
  else echo 32768
  fi
}

TRAIN_BATCH_TOKENS_AUTO="$(pick_train_batch_tokens "${FREE_MB_MIN}")"
# Keep eval deterministic and OOM-safe for submission pipeline.
if (( FREE_MB_MIN >= 50000 )); then
  SLIDING_BATCH_SIZE_AUTO=128
else
  SLIDING_BATCH_SIZE_AUTO=64
fi

LOG_FILE="logs/max_vram_10min/${RUN_ID}.log"

COMMON_ENV=(
  RUN_ID="${RUN_ID}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  DDP_FIND_UNUSED_PARAMETERS=1
  OMP_NUM_THREADS=8
  DATA_PATH=./data/datasets/fineweb10B_sp8192
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
  COMPETITION_PROFILE=1
  EXPORT_MODE=competition_gptq
  RUNTIME_PATH_POLICY=strict
  HARD_BUDGET_BYTES=16000000
  HARD_BUDGET_ENFORCE=1
  MAX_WALLCLOCK_SECONDS=599
  ITERATIONS=200000
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-${TRAIN_BATCH_TOKENS_AUTO}}"
  TRAIN_SEQ_LEN=1024
  SLIDING_EVAL=1
  SLIDING_BATCH_SIZE="${SLIDING_BATCH_SIZE:-${SLIDING_BATCH_SIZE_AUTO}}"
  FINAL_EVAL_SEQUENTIAL_CARRY=1
  COMPILE_MODE=max-autotune
  COMPILER_WARMUP_STEPS=1
  SYNTHETIC_WARMUP=1
  TORCHINDUCTOR_FX_GRAPH_CACHE=1
  TORCHINDUCTOR_AUTOGRAD_CACHE=1
  COMPILE_SHAPE_PADDING=1
  COMPILE_TRITON_CUDAGRAPHS=0
  TERNARY_THRESHOLD_SEARCH=1
  TERNARY_SCALE_SEARCH=1
  EXPORT_ALIGNED_TRAIN=1
  EXPORT_ALIGNED_TRAIN_START_FRACTION=0.75
  EXPORT_PROXY_EVAL=1
  EXPORT_PROXY_EVERY=250
  EXPORT_PROXY_NUM_SEQS=16
  TERNARY_COMPRESS_BROTLI=1
  ENGRAM_COMPETITION_ENABLED="${ENGRAM_COMPETITION_ENABLED:-1}"
  ENGRAM_EXPORT_PRUNE_ENABLED=1
  ENGRAM_EXPORT_KEEP_BIGRAM_RATIO="${ENGRAM_EXPORT_KEEP_BIGRAM_RATIO:-0.45}"
  ENGRAM_EXPORT_KEEP_TRIGRAM_RATIO="${ENGRAM_EXPORT_KEEP_TRIGRAM_RATIO:-0.20}"
  ENGRAM_EXPORT_KEEP_MIN_BUCKETS="${ENGRAM_EXPORT_KEEP_MIN_BUCKETS:-256}"
  ENGRAM_EXPORT_SCORE_ALPHA="${ENGRAM_EXPORT_SCORE_ALPHA:-0.80}"
  ENGRAM_EXPORT_TOKEN_BUDGET="${ENGRAM_EXPORT_TOKEN_BUDGET:-32768}"
)

echo "run_id=${RUN_ID} nproc=${NPROC} free_mb_min=${FREE_MB_MIN} train_batch_tokens=${TRAIN_BATCH_TOKENS_AUTO} sliding_batch=${SLIDING_BATCH_SIZE_AUTO}" | tee "${LOG_FILE}"

env "${COMMON_ENV[@]}" python3 build_submission.py >/dev/null

env "${COMMON_ENV[@]}" torchrun --standalone --nproc_per_node="${NPROC}" train_gpt.py 2>&1 | tee -a "${LOG_FILE}"

echo "Done. Log: ${LOG_FILE}"
