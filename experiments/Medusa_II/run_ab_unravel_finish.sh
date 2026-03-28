#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

AB_INIT_MODEL_PATH="${AB_INIT_MODEL_PATH:-final_model.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
AB_CASES="${AB_CASES:-A,B,C}"
AB_GPTQ_CALIB_SAMPLES="${AB_GPTQ_CALIB_SAMPLES:-64}"
AB_GPTQ_CALIB_SEQ_LEN="${AB_GPTQ_CALIB_SEQ_LEN:-1024}"
AB_VAL_TOKEN_CAP="${AB_VAL_TOKEN_CAP:-62021632}"

if [[ ! -f "${AB_INIT_MODEL_PATH}" ]]; then
  echo "Missing init model: ${AB_INIT_MODEL_PATH}"
  echo "Set AB_INIT_MODEL_PATH=/path/to/model.pt and re-run."
  exit 2
fi

env \
  AB_INIT_MODEL_PATH="${AB_INIT_MODEL_PATH}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  AB_CASES="${AB_CASES}" \
  AB_GPTQ_CALIB_SAMPLES="${AB_GPTQ_CALIB_SAMPLES}" \
  AB_GPTQ_CALIB_SEQ_LEN="${AB_GPTQ_CALIB_SEQ_LEN}" \
  AB_VAL_TOKEN_CAP="${AB_VAL_TOKEN_CAP}" \
  bash "${SCRIPT_DIR}/run_ab_unravel_short.sh"
