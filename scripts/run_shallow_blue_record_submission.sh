#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="${SUBMISSION_DIR:-records/track_10min_16mb/2026-04-07_Shallow_Blue_Probe_BOS}"
RUN_ID="${RUN_ID:-shallow_blue_record_submission_$(date +%Y%m%d_%H%M%S)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-595}"
PYTHON_BIN="${PYTHON:-python3}"
DATA_PATH="${DATA_PATH:-../../../data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-../../../data/tokenizers/fineweb_1024_bpe.model}"
SHALLOW_BLUE_EVAL_ENABLED="${SHALLOW_BLUE_EVAL_ENABLED:-1}"
SHALLOW_BLUE_WINDOW="${SHALLOW_BLUE_WINDOW:-1024}"
SHALLOW_BLUE_STRIDE="${SHALLOW_BLUE_STRIDE:-1024}"
SHALLOW_BLUE_BATCH_WINDOWS="${SHALLOW_BLUE_BATCH_WINDOWS:-32}"
SHALLOW_BLUE_ALPHA="${SHALLOW_BLUE_ALPHA:-0.30}"
SHALLOW_BLUE_MAX_DOCS="${SHALLOW_BLUE_MAX_DOCS:-0}"
SHALLOW_BLUE_MAX_VAL_TOKENS="${SHALLOW_BLUE_MAX_VAL_TOKENS:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${SUBMISSION_DIR}" ]]; then
  echo "Submission dir not found: ${SUBMISSION_DIR}" >&2
  exit 1
fi

pushd "${SUBMISSION_DIR}" >/dev/null

mkdir -p logs

CMD=(
  env
  OMP_NUM_THREADS=1
  PYTHONFAULTHANDLER=1
  RUN_ID="${RUN_ID}"
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}"
  DATA_PATH="${DATA_PATH}"
  TOKENIZER_PATH="${TOKENIZER_PATH}"
  SHALLOW_BLUE_EVAL_ENABLED="${SHALLOW_BLUE_EVAL_ENABLED}"
  SHALLOW_BLUE_WINDOW="${SHALLOW_BLUE_WINDOW}"
  SHALLOW_BLUE_STRIDE="${SHALLOW_BLUE_STRIDE}"
  SHALLOW_BLUE_BATCH_WINDOWS="${SHALLOW_BLUE_BATCH_WINDOWS}"
  SHALLOW_BLUE_ALPHA="${SHALLOW_BLUE_ALPHA}"
  SHALLOW_BLUE_MAX_DOCS="${SHALLOW_BLUE_MAX_DOCS}"
  SHALLOW_BLUE_MAX_VAL_TOKENS="${SHALLOW_BLUE_MAX_VAL_TOKENS}"
  "${PYTHON_BIN}"
  -m
  torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  train_gpt.py
)

echo "Shallow Blue record submission launcher"
echo "  submission_dir:       ${SUBMISSION_DIR}"
echo "  run_id:               ${RUN_ID}"
echo "  nproc_per_node:       ${NPROC_PER_NODE}"
echo "  max_wallclock_sec:    ${MAX_WALLCLOCK_SECONDS}"
echo "  data_path:            ${DATA_PATH}"
echo "  tokenizer_path:       ${TOKENIZER_PATH}"
echo "  shallow_blue_window/stride: ${SHALLOW_BLUE_WINDOW}/${SHALLOW_BLUE_STRIDE}"
echo "  shallow_blue_batch_windows: ${SHALLOW_BLUE_BATCH_WINDOWS}"
echo "  shallow_blue_alpha:         ${SHALLOW_BLUE_ALPHA}"
echo "  shallow_blue_max_docs:      ${SHALLOW_BLUE_MAX_DOCS}"
echo "  shallow_blue_max_val_tokens:${SHALLOW_BLUE_MAX_VAL_TOKENS}"
printf '  command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  popd >/dev/null
  exit 0
fi

"${CMD[@]}"

if [[ -f "logs/${RUN_ID}.txt" ]]; then
  cp "logs/${RUN_ID}.txt" latest_run.log
fi
printf '%s\n' "${RUN_ID}" > latest_run.id.txt
"${PYTHON_BIN}" ../../../scripts/print_shallow_blue_record_submission_summary.py . --json > latest_run.summary.json
"${PYTHON_BIN}" ../../../scripts/print_shallow_blue_record_submission_summary.py .

popd >/dev/null
