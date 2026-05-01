#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DATASET_DIR="${BASE_DATASET_DIR:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
DATA_VIEW_DIR="${DATA_VIEW_DIR:-${SCRIPT_DIR}/runtime_data/fineweb10B_sp1024_10shards}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"

if [[ "${TRAIN_SHARDS}" != "10" ]]; then
  echo "run.sh is pinned to 10 train shards for this Colab replica. Got TRAIN_SHARDS=${TRAIN_SHARDS}." >&2
  exit 1
fi

mkdir -p "${DATA_VIEW_DIR}"
cd "${SCRIPT_DIR}"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt"
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
  export PG_COLAB_FORCE_FP16="${PG_COLAB_FORCE_FP16:-0}"
else
  export PG_COLAB_FORCE_FP16="${PG_COLAB_FORCE_FP16:-1}"
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

export DATA_PATH="${DATA_PATH:-${DATA_VIEW_DIR}}"
export TOKENIZER_PATH
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
export BIGRAM_DIM="${BIGRAM_DIM:-112}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}"
export TARGET_MB="${TARGET_MB:-15.9}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export ITERATIONS="${ITERATIONS:-20000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export SEED="${SEED:-314}"
export PG_COLAB_DISABLE_COMPILE="${PG_COLAB_DISABLE_COMPILE:-1}"
export PG_COLAB_DISABLE_FUSED_ADAM="${PG_COLAB_DISABLE_FUSED_ADAM:-0}"

exec python3 "${SCRIPT_DIR}/train_gpt.py"
