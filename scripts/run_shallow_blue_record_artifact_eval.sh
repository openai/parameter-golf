#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-shallow_blue_record_artifact_eval_$(date +%Y%m%d_%H%M%S)}"
SUBMISSION_DIR="${SUBMISSION_DIR:-records/track_10min_16mb/2026-04-07_Shallow_Blue_Probe_BOS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODEL_PATH="${MODEL_PATH:-${SUBMISSION_DIR}/final_model.int8.ptz}"
PROBE_ARTIFACT="${PROBE_ARTIFACT:-${SUBMISSION_DIR}/shallow_blue_probe.json}"
VAL_FILES="${VAL_FILES:-data/datasets/fineweb10B_sp1024/fineweb_val_*.bin}"
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizers/fineweb_1024_bpe.model}"
WINDOW="${WINDOW:-1024}"
STRIDE="${STRIDE:-1024}"
BATCH_WINDOWS="${BATCH_WINDOWS:-32}"
ALPHA="${ALPHA:-0.30}"
MAX_DOCS="${MAX_DOCS:-0}"
MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-0}"
PYTHON_BIN="${PYTHON:-python3}"
LOG_PATH="${LOG_PATH:-reports/shallow_blue/${RUN_ID}.log}"

if [[ ! -d "${SUBMISSION_DIR}" ]]; then
  echo "Submission directory not found: ${SUBMISSION_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Artifact not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PROBE_ARTIFACT}" ]]; then
  echo "Probe artifact not found: ${PROBE_ARTIFACT}" >&2
  exit 1
fi
mkdir -p "$(dirname "${LOG_PATH}")"

CMD=(
  env
  OMP_NUM_THREADS=1
  PYTHONFAULTHANDLER=1
  "${PYTHON_BIN}"
  -m
  torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  scripts/eval_shallow_blue_record_artifact.py
  --submission-dir "${SUBMISSION_DIR}"
  --model-path "${MODEL_PATH}"
  --probe-artifact "${PROBE_ARTIFACT}"
  --val-files "${VAL_FILES}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --window "${WINDOW}"
  --stride "${STRIDE}"
  --batch-windows "${BATCH_WINDOWS}"
  --alpha "${ALPHA}"
  --max-docs "${MAX_DOCS}"
  --max-val-tokens "${MAX_VAL_TOKENS}"
)

echo "Shallow Blue record artifact evaluator"
echo "  run_id:         ${RUN_ID}"
echo "  submission_dir: ${SUBMISSION_DIR}"
echo "  nproc:          ${NPROC_PER_NODE}"
echo "  model:          ${MODEL_PATH}"
echo "  probe:          ${PROBE_ARTIFACT}"
echo "  val files:      ${VAL_FILES}"
echo "  tokenizer:      ${TOKENIZER_PATH}"
echo "  window/stride:  ${WINDOW}/${STRIDE}"
echo "  batch windows:  ${BATCH_WINDOWS}"
echo "  alpha:          ${ALPHA}"
echo "  max docs:       ${MAX_DOCS}"
echo "  max tokens:     ${MAX_VAL_TOKENS}"
echo "  log:            ${LOG_PATH}"
printf '  command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" | tee "${LOG_PATH}"
