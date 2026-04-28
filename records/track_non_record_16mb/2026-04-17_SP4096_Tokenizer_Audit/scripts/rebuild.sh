#!/usr/bin/env bash
# Single-command rebuild for the 4096-vocab SentencePiece BPE tokenizer.
# Runs the wrapper and tees all output to logs/build.log.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
find_repo_root() {
  local current="$1"
  while [[ "${current}" != "/" ]]; do
    if [[ -f "${current}/data/download_hf_docs_and_tokenize.py" ]]; then
      printf '%s\n' "${current}"
      return 0
    fi
    current="$(dirname "${current}")"
  done
  return 1
}
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(find_repo_root "${ROOT}")"
WRAPPER="${SCRIPT_DIR}/build_sp4096.py"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${TOKENIZER_4K_LOG_FILE:-${LOG_DIR}/build.log}"
DEFAULT_OUTPUT_ROOT="${TOKENIZER_4K_DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${MATCHED_FINEWEB_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
DOCS_PATH="${TOKENIZER_4K_DOCS_PATH:-}"
EXPECTED_DOCS_SHA="${TOKENIZER_4K_EXPECTED_DOCS_SHA:-}"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

export MATCHED_FINEWEB_TOKENIZER_THREADS="$(nproc)"
export MATCHED_FINEWEB_SP_BATCH_SIZE=4096

EXTRA_ARGS=()
if [[ -n "${DOCS_PATH}" ]]; then
  EXTRA_ARGS+=(--docs-path "${DOCS_PATH}")
fi
if [[ -n "${EXPECTED_DOCS_SHA}" ]]; then
  EXTRA_ARGS+=(--expected-docs-sha "${EXPECTED_DOCS_SHA}")
fi

echo "rebuild.sh start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_FILE}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}" | tee -a "${LOG_FILE}"
echo "DOCS_PATH=${DOCS_PATH:-<default>}" | tee -a "${LOG_FILE}"
echo "MATCHED_FINEWEB_TOKENIZER_THREADS=${MATCHED_FINEWEB_TOKENIZER_THREADS}" | tee -a "${LOG_FILE}"
echo "MATCHED_FINEWEB_SP_BATCH_SIZE=${MATCHED_FINEWEB_SP_BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "REPO_ROOT=${REPO_ROOT}" | tee -a "${LOG_FILE}"
echo "wrapper: ${WRAPPER}" | tee -a "${LOG_FILE}"

python3 -u "${WRAPPER}" --output-root "${OUTPUT_ROOT}" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
rc=${PIPESTATUS[0]}

echo "rebuild.sh end: $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc}" | tee -a "${LOG_FILE}"
exit "${rc}"
