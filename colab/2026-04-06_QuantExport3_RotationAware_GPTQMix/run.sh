#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DATASET_DIR="${BASE_DATASET_DIR:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
DATA_VIEW_DIR="${DATA_VIEW_DIR:-${SCRIPT_DIR}/runtime_data/fineweb10B_sp1024_10shards}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"

if [[ "${TRAIN_SHARDS}" != "10" ]]; then
  echo "run.sh is pinned to 10 train shards for benchmark comparability. Got TRAIN_SHARDS=${TRAIN_SHARDS}." >&2
  exit 1
fi

mkdir -p "${DATA_VIEW_DIR}"
cd "${SCRIPT_DIR}"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  python3 -m pip install -r "${REPO_ROOT}/requirements.txt" -r "${SCRIPT_DIR}/requirements.txt"
fi

if [[ "${DOWNLOAD_DATA:-1}" == "1" ]]; then
  python3 "${REPO_ROOT}/data/cached_challenge_fineweb.py" --variant sp1024 --train-shards 10
fi

for idx in $(seq 0 9); do
  shard_name="$(printf 'fineweb_train_%06d.bin' "${idx}")"
  src="${BASE_DATASET_DIR}/${shard_name}"
  dst="${DATA_VIEW_DIR}/${shard_name}"
  if [[ ! -f "${src}" ]]; then
    echo "Missing expected train shard: ${src}" >&2
    exit 1
  fi
  ln -sfn "${src}" "${dst}"
done

for src in "${BASE_DATASET_DIR}"/fineweb_val_*.bin; do
  if [[ ! -f "${src}" ]]; then
    echo "Validation shards were not found under ${BASE_DATASET_DIR}" >&2
    exit 1
  fi
  ln -sfn "${src}" "${DATA_VIEW_DIR}/$(basename "${src}")"
done

if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)"; then
  export COMPUTE_DTYPE="${COMPUTE_DTYPE:-bf16}"
  export MUON_DTYPE="${MUON_DTYPE:-bf16}"
else
  export COMPUTE_DTYPE="${COMPUTE_DTYPE:-fp16}"
  export MUON_DTYPE="${MUON_DTYPE:-fp16}"
fi

if [[ -z "${EXPORT_ARTIFACT_NAME:-}" ]]; then
  if [[ "${EXPORT_COMPRESSOR:-lzma}" == "zlib" ]]; then
    export EXPORT_ARTIFACT_NAME="final_model.int8.ptz"
  else
    export EXPORT_ARTIFACT_NAME="final_model.int8.ptx"
  fi
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

export DATA_PATH="${DATA_PATH:-${DATA_VIEW_DIR}}"
export TOKENIZER_PATH
export RUN_ID="${RUN_ID:-2026-04-06_quant_export3_rotation_gptqmix}"

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export ITERATIONS="${ITERATIONS:-20000}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export SEED="${SEED:-314}"
export ENABLE_COMPILE="${ENABLE_COMPILE:-0}"
export ENABLE_FUSED_ADAM="${ENABLE_FUSED_ADAM:-0}"
export ENABLE_MATH_SDP="${ENABLE_MATH_SDP:-1}"

export EXPORT_COMPRESSOR="${EXPORT_COMPRESSOR:-lzma}"
export EXPORT_COMPRESSION_LEVEL="${EXPORT_COMPRESSION_LEVEL:-9}"
export INT8_CLIP_PERCENTILE="${INT8_CLIP_PERCENTILE:-99.99995}"
export INT8_KEEP_FLOAT_MAX_NUMEL="${INT8_KEEP_FLOAT_MAX_NUMEL:-131072}"
export INT8_KEEP_FLOAT_STORE_DTYPE="${INT8_KEEP_FLOAT_STORE_DTYPE:-fp16}"
export INT8_PER_ROW_SCALE_DTYPE="${INT8_PER_ROW_SCALE_DTYPE:-fp16}"
export GPTQ_BLOCK_SIZE="${GPTQ_BLOCK_SIZE:-128}"
export ROTATION_AWARE_ENABLED="${ROTATION_AWARE_ENABLED:-1}"
export ROTATION_BLOCK_SIZE="${ROTATION_BLOCK_SIZE:-128}"
export HESSIAN_DAMPING="${HESSIAN_DAMPING:-0.01}"
export GPTQ_ACCEPT_MSE_RATIO="${GPTQ_ACCEPT_MSE_RATIO:-1.05}"
export MIXED_PRECISION_EXTRA_BUDGET_BYTES="${MIXED_PRECISION_EXTRA_BUDGET_BYTES:-524288}"
export MIXED_PRECISION_MAX_TENSORS="${MIXED_PRECISION_MAX_TENSORS:-2}"
export SEARCH_ROTATION_OPTIONS="${SEARCH_ROTATION_OPTIONS:-0,1}"
export SEARCH_MIXED_PRECISION_BUDGETS="${SEARCH_MIXED_PRECISION_BUDGETS:-0,524288}"
export SEARCH_MIXED_PRECISION_MAX_TENSORS="${SEARCH_MIXED_PRECISION_MAX_TENSORS:-0,2}"
export SEARCH_TARGET_TOTAL_BYTES="${SEARCH_TARGET_TOTAL_BYTES:-16777216}"
export SEARCH_MAX_FRONTIER_EVALS="${SEARCH_MAX_FRONTIER_EVALS:-6}"
export SELF_CALIB_VARIANTS="${SELF_CALIB_VARIANTS:-ar_sample,ar_greedy,ar_topk}"
export SELF_CALIB_NUM_SEQS="${SELF_CALIB_NUM_SEQS:-24}"
export SELF_CALIB_SEQ_LEN="${SELF_CALIB_SEQ_LEN:-512}"
export SELF_CALIB_BATCH_SIZE="${SELF_CALIB_BATCH_SIZE:-4}"
export SELF_CALIB_TEMPERATURE="${SELF_CALIB_TEMPERATURE:-0.8}"
export SELF_CALIB_TOPK="${SELF_CALIB_TOPK:-32}"
export SELF_CALIB_CANDIDATE_PERCENTILES="${SELF_CALIB_CANDIDATE_PERCENTILES:-99.999,99.99984,99.99995,100.0}"

exec python3 "${SCRIPT_DIR}/train_gpt.py"
