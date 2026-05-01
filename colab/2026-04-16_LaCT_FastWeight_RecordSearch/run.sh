#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
DATASET_DIR="${DATASET_DIR:-${DATA_DIR}/datasets/fineweb10B_sp8192}"
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-kevclark/parameter-golf}"

mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

if [[ "${DOWNLOAD_DATA:-1}" == "1" ]]; then
  MANIFEST_PATH="${DATA_DIR}/manifest.json"
  if [[ -f "${MANIFEST_PATH}" ]] && ! python3 - "${MANIFEST_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    manifest = json.load(f)
sys.exit(0 if any(x.get("name") == "fineweb10B_sp8192" for x in manifest.get("datasets", [])) else 1)
PY
  then
    echo "Refreshing cached manifest without fineweb10B_sp8192: ${MANIFEST_PATH}" >&2
    rm -f "${MANIFEST_PATH}"
  fi
  python3 "${REPO_ROOT}/data/cached_challenge_fineweb.py" --variant sp8192 --train-shards "${TRAIN_SHARDS}"
fi

if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)"; then
  export COMPUTE_DTYPE="${COMPUTE_DTYPE:-bf16}"
else
  export COMPUTE_DTYPE="${COMPUTE_DTYPE:-fp16}"
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

export DATA_DIR
export RUN_ID="${RUN_ID:-2026-04-16_lact_fastweight_record_search}"
export SEED="${SEED:-42}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-5.25}"

# LaCT is the primary record path here. Leave legacy TTT off unless comparing.
export LACT_TTT_ENABLED="${LACT_TTT_ENABLED:-1}"
export TTT_ENABLED="${TTT_ENABLED:-0}"
export TTT_LR="${TTT_LR:-0.005}"
export TTT_EPOCHS="${TTT_EPOCHS:-3}"
export TTT_DOC_LOCAL="${TTT_DOC_LOCAL:-0}"
export TTT_DOC_BOUNDARY_TOKEN="${TTT_DOC_BOUNDARY_TOKEN:-0}"
export LACT_FAST_WEIGHT="${LACT_FAST_WEIGHT:-swiglu}"
export LACT_UPDATE="${LACT_UPDATE:-muon}"
export LACT_CHUNK_TOKENS="${LACT_CHUNK_TOKENS:-32768}"
export LACT_LR="${LACT_LR:-0.02}"
export LACT_MOMENTUM="${LACT_MOMENTUM:-0.9}"
export LACT_EPOCHS="${LACT_EPOCHS:-1}"
export LACT_SCALE="${LACT_SCALE:-0.08}"
export LACT_GRAD_CLIP="${LACT_GRAD_CLIP:-1.0}"
export LACT_NORMALIZE="${LACT_NORMALIZE:-1}"

# Single-GPU defaults are for smoke tests. RECORD_PROFILE=1 is the intended
# 8xH100 SXM record attempt profile.
if [[ "${RECORD_PROFILE:-0}" == "1" ]]; then
  export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
  export VAL_BATCH_TOKENS="${VAL_BATCH_TOKENS:-524288}"
  export GPTQ_CALIBRATION_BATCHES="${GPTQ_CALIBRATION_BATCHES:-64}"
  export ENABLE_COMPILE="${ENABLE_COMPILE:-1}"
  export LACT_STATE_DIM="${LACT_STATE_DIM:-128}"
  export LACT_BATCH_SEQS="${LACT_BATCH_SEQS:-16}"
  export LACT_BASE_TTT="${LACT_BASE_TTT:-1}"
else
  export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
  export VAL_BATCH_TOKENS="${VAL_BATCH_TOKENS:-65536}"
  export GPTQ_CALIBRATION_BATCHES="${GPTQ_CALIBRATION_BATCHES:-16}"
  export GPTQ_RESERVE_SECONDS="${GPTQ_RESERVE_SECONDS:-90}"
  export ENABLE_COMPILE="${ENABLE_COMPILE:-0}"
  export PG_COLAB_DISABLE_COMPILE="${PG_COLAB_DISABLE_COMPILE:-1}"
  export LACT_STATE_DIM="${LACT_STATE_DIM:-64}"
  export LACT_BATCH_SEQS="${LACT_BATCH_SEQS:-4}"
  export LACT_BASE_TTT="${LACT_BASE_TTT:-0}"
fi

export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export ITERATIONS="${ITERATIONS:-20000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export COMPRESSOR="${COMPRESSOR:-brotli}"
export EXPORT_ALLOCATOR="${EXPORT_ALLOCATOR:-entropy}"
export ARTIFACT_TARGET_BYTES="${ARTIFACT_TARGET_BYTES:-16000000}"
export ALLOCATOR_GROUP_COLS="${ALLOCATOR_GROUP_COLS:-128}"
export ALLOCATOR_MATRIX_BITS="${ALLOCATOR_MATRIX_BITS:-5,6,7}"
export ALLOCATOR_MLP_BITS="${ALLOCATOR_MLP_BITS:-}"
export ALLOCATOR_ATTN_BITS="${ALLOCATOR_ATTN_BITS:-}"
export ALLOCATOR_EMBED_BITS="${ALLOCATOR_EMBED_BITS:-7,8}"
export ALLOCATOR_MATRIX_SIGMAS="${ALLOCATOR_MATRIX_SIGMAS:-10.5,12.85,15.0}"
export ALLOCATOR_MLP_SIGMAS="${ALLOCATOR_MLP_SIGMAS:-}"
export ALLOCATOR_ATTN_SIGMAS="${ALLOCATOR_ATTN_SIGMAS:-}"
export ALLOCATOR_EMBED_SIGMAS="${ALLOCATOR_EMBED_SIGMAS:-16.0,20.0,24.0}"
export ALLOCATOR_USE_ENTROPY_PROXY="${ALLOCATOR_USE_ENTROPY_PROXY:-1}"
export ALLOCATOR_CODE_WRAPPERS="${ALLOCATOR_CODE_WRAPPERS:-source,lzma_raw_b85_exec}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing dataset directory: ${DATASET_DIR}" >&2
  exit 1
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/train_gpt.py"
fi

exec python3 "${SCRIPT_DIR}/train_gpt.py"
