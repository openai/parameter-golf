#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
MIN_GPUS="${MIN_GPUS:-8}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: expected python at ${PYTHON_BIN}" >&2
  echo "hint: create the repo venv first or override PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/eval"

echo "record_dir=${SCRIPT_DIR}"
echo "repo_root=${REPO_ROOT}"
echo "python_bin=${PYTHON_BIN}"
echo "data_path=${DATA_PATH}"
echo "tokenizer_path=${TOKENIZER_PATH}"
echo "min_gpus=${MIN_GPUS}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/eval/preflight.py" \
  --data-path "${DATA_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --min-gpus "${MIN_GPUS}"

echo
echo "prepare.sh: environment looks ready"
echo "next: SEED=42 bash eval/eval.sh"
