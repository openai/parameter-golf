#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECORD_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${RECORD_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${REPO_ROOT}/.venv/bin/torchrun}"

SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-train_seed${SEED}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
TRAIN_LOG_NAME="${TRAIN_LOG_NAME:-train_seed${SEED}.log}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: expected python at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -x "${TORCHRUN_BIN}" ]]; then
  LAUNCHER=("${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py)
else
  LAUNCHER=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py)
fi

echo "record_dir=${RECORD_DIR}"
echo "repo_root=${REPO_ROOT}"
echo "seed=${SEED}"
echo "run_id=${RUN_ID}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "data_path=${DATA_PATH}"
echo "tokenizer_path=${TOKENIZER_PATH}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" \
  --data-path "${DATA_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --min-gpus "${NPROC_PER_NODE}"

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "precheck_only=1"
  echo "launcher=${LAUNCHER[*]}"
  exit 0
fi

cd "${RECORD_DIR}"
mkdir -p logs

env \
  RUN_ID="${RUN_ID}" \
  SEED="${SEED}" \
  DATA_PATH="${DATA_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  VOCAB_SIZE="${VOCAB_SIZE}" \
  "${LAUNCHER[@]}"

SOURCE_LOG="${RECORD_DIR}/logs/${RUN_ID}.txt"
TARGET_LOG="${RECORD_DIR}/${TRAIN_LOG_NAME}"

if [[ ! -f "${SOURCE_LOG}" ]]; then
  echo "error: expected train log at ${SOURCE_LOG}" >&2
  exit 1
fi

cp "${SOURCE_LOG}" "${TARGET_LOG}"
echo "canonical_log=${TARGET_LOG}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/parse_metrics.py" "${TARGET_LOG}"
