#!/usr/bin/env bash
set -euo pipefail

# Record-local data setup for this submission.
# Exports SP8192 dataset shards (80 train shards) into this record folder.
#
# Usage:
#   bash ./setup_sp8192_data.sh
# Optional:
#   HF_TOKEN=... bash ./setup_sp8192_data.sh
#   VENV_DIR=.venv bash ./setup_sp8192_data.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VOCAB_SIZE="${VOCAB_SIZE:-8192}"
MAX_TRAIN_SHARDS="${MAX_TRAIN_SHARDS:-80}"
VENV_DIR="${VENV_DIR:-.venv}"
OUTPUT_ROOT="${SCRIPT_DIR}/sp8192_data"
TOKENIZER_MODEL="${SCRIPT_DIR}/fineweb_8192_bpe.model"
REQ_FILE="${SCRIPT_DIR}/reqs.txt"

if [[ ! -f "${REPO_ROOT}/build_sp_dataset.sh" ]]; then
  echo "ERROR: build_sp_dataset.sh not found at repo root: ${REPO_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${TOKENIZER_MODEL}" ]]; then
  echo "ERROR: tokenizer model not found: ${TOKENIZER_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[setup] repo_root=${REPO_ROOT}"
echo "[setup] output_root=${OUTPUT_ROOT}"
echo "[setup] vocab_size=${VOCAB_SIZE} max_train_shards=${MAX_TRAIN_SHARDS}"
echo "[setup] tokenizer_reuse=${TOKENIZER_MODEL}"
echo "[setup] venv=${VENV_DIR}"
echo "[setup] reqs=${REQ_FILE}"

# Bootstrap venv + deps if needed so this script is self-contained.
if [[ ! -f "${REPO_ROOT}/${VENV_DIR}/bin/activate" ]]; then
  echo "[setup] creating virtualenv at ${REPO_ROOT}/${VENV_DIR}"
  python3 -m venv "${REPO_ROOT}/${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${REPO_ROOT}/${VENV_DIR}/bin/activate"
python3 -m pip install --upgrade pip wheel setuptools >/dev/null
python3 -m pip install -r "${REQ_FILE}"

cd "${REPO_ROOT}"

VOCAB_SIZE="${VOCAB_SIZE}" \
VENV_DIR="${VENV_DIR}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
MAX_TRAIN_SHARDS="${MAX_TRAIN_SHARDS}" \
EXISTING_TOKENIZER_MODEL="${TOKENIZER_MODEL}" \
bash ./build_sp_dataset.sh

echo
echo "[done] dataset path:"
echo "  ${OUTPUT_ROOT}/datasets/fineweb10B_sp8192"
echo "[done] train shards count:"
find "${OUTPUT_ROOT}/datasets/fineweb10B_sp8192" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l
