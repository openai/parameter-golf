#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

AB_INIT_MODEL_PATH="${AB_INIT_MODEL_PATH:-final_model.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SEED="${SEED:-1337}"
AB_GPTQ_CALIB_SAMPLES="${AB_GPTQ_CALIB_SAMPLES:-64}"
AB_GPTQ_CALIB_SEQ_LEN="${AB_GPTQ_CALIB_SEQ_LEN:-1024}"
AB_VAL_TOKEN_CAP="${AB_VAL_TOKEN_CAP:-62021632}"

if [[ ! -f "${AB_INIT_MODEL_PATH}" ]]; then
  echo "Missing init model: ${AB_INIT_MODEL_PATH}"
  echo "Set AB_INIT_MODEL_PATH=/path/to/model.pt and re-run."
  exit 2
fi

mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo "  MEDUSA_II additional-only unravel checks"
echo "  additive mode: does NOT rerun A/B/C GPTQ"
echo "  seed=${SEED} nproc=${NPROC_PER_NODE}"
echo "  init_model_path=${AB_INIT_MODEL_PATH}"
echo "============================================"

run_case() {
  local label="$1"
  shift
  local log="logs/medusa2_ab_additional_${label}_s${SEED}_${ts}.log"
  local summary_pattern="gptq:calibration config|gptq:SKIPPED|gptq:loop-aware|final_int6_roundtrip_exact|final_int6_sliding_window_exact|final_int8_zlib_roundtrip_exact"

  echo
  echo ">>> ADDITIONAL CASE ${label}"
  echo ">>> LOG ${log}"

  env \
    SEED="${SEED}" \
    AB_INIT_MODEL_PATH="${AB_INIT_MODEL_PATH}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    AB_GPTQ_CALIB_SAMPLES="${AB_GPTQ_CALIB_SAMPLES}" \
    AB_GPTQ_CALIB_SEQ_LEN="${AB_GPTQ_CALIB_SEQ_LEN}" \
    AB_VAL_TOKEN_CAP="${AB_VAL_TOKEN_CAP}" \
    "$@" \
    bash "${SCRIPT_DIR}/run_ab_unravel_finish.sh" \
    2>&1 | tee "${log}"

  echo ">>> SUMMARY ${label}"
  if command -v rg >/dev/null 2>&1; then
    rg -n "${summary_pattern}" "${log}" || true
  else
    grep -nE "${summary_pattern}" "${log}" || true
  fi
}

# Missing check 1: loop-aware OFF (D) against your existing A/B/C run.
run_case "D_no_loopaware" AB_CASES=D AB_INCLUDE_NO_LOOP_AWARE=1 SKIP_GPTQ=0

# Missing check 2: no GPTQ control (naive int6 quantization).
run_case "A_skip_gptq" AB_CASES=A AB_INCLUDE_NO_LOOP_AWARE=0 SKIP_GPTQ=1

echo
echo "============================================"
echo "  ADDITIONAL-ONLY CHECKS DONE"
echo "============================================"
