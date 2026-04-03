#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RECORD_DIR="${ROOT}/records/track_non_record_16mb/2026-04-03_DeepFloor"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TMP_DIR="${TMP_DIR:-$(mktemp -d /tmp/deepfloor-submission-preflight.XXXXXX)}"
RESULT_JSON="${RESULT_JSON:-${TMP_DIR}/train_result.json}"
SUBMISSION_JSON="${SUBMISSION_JSON:-${TMP_DIR}/submission.json}"
TRAIN_STEPS="${TRAIN_STEPS:-1}"
EVAL_BATCHES="${EVAL_BATCHES:-1}"
SEQ_LEN="${SEQ_LEN:-16}"
STRIDE="${STRIDE:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RECURRENT_DIM="${RECURRENT_DIM:-32}"
NUM_DISTINCT_BLOCKS="${NUM_DISTINCT_BLOCKS:-1}"
VIEW_COUNT="${VIEW_COUNT:-2}"
TRAIN_RECURRENCE_STEPS="${TRAIN_RECURRENCE_STEPS:-4}"
EVAL_RECURRENCE_STEPS="${EVAL_RECURRENCE_STEPS:-6}"
DEVICE="${DEVICE:-cpu}"

if [[ -n "${ENWIK8_PATH:-}" ]]; then
  INPUT_ENWIK8="${ENWIK8_PATH}"
else
  INPUT_ENWIK8="${TMP_DIR}/enwik8"
  dd if=/dev/zero of="${INPUT_ENWIK8}" bs=32768 count=1 status=none
fi

echo "[preflight] tmp_dir=${TMP_DIR}"
echo "[preflight] result_json=${RESULT_JSON}"
echo "[preflight] submission_json=${SUBMISSION_JSON}"

"${PYTHON_BIN}" -m py_compile \
  "${RECORD_DIR}/freeze_submission_snapshot.py" \
  "${RECORD_DIR}/deepfloor_snapshot.py" \
  "${RECORD_DIR}/train_gpt.py" \
  "${RECORD_DIR}/build_submission_json.py"

ENWIK8_PATH="${INPUT_ENWIK8}" \
OUTPUT_JSON="${RESULT_JSON}" \
DEVICE="${DEVICE}" \
TRAIN_STEPS="${TRAIN_STEPS}" \
EVAL_BATCHES="${EVAL_BATCHES}" \
SEQ_LEN="${SEQ_LEN}" \
STRIDE="${STRIDE}" \
BATCH_SIZE="${BATCH_SIZE}" \
RECURRENT_DIM="${RECURRENT_DIM}" \
NUM_DISTINCT_BLOCKS="${NUM_DISTINCT_BLOCKS}" \
VIEW_COUNT="${VIEW_COUNT}" \
TRAIN_RECURRENCE_STEPS="${TRAIN_RECURRENCE_STEPS}" \
EVAL_RECURRENCE_STEPS="${EVAL_RECURRENCE_STEPS}" \
CACHE_DATASET_ON_DEVICE=0 \
"${PYTHON_BIN}" "${RECORD_DIR}/train_gpt.py"

"${PYTHON_BIN}" "${RECORD_DIR}/build_submission_json.py" \
  --result-json "${RESULT_JSON}" \
  --output-json "${SUBMISSION_JSON}"

"${PYTHON_BIN}" - <<'PY' "${RESULT_JSON}" "${SUBMISSION_JSON}"
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
submission = json.loads(Path(sys.argv[2]).read_text())
artifact_bytes = int(result["artifact"]["estimated_bytes"])
total_bytes = int(submission["bytes_total"])
if artifact_bytes <= 0:
    raise SystemExit("artifact_bytes must be positive")
if total_bytes >= 16_000_000:
    raise SystemExit(f"submission exceeds 16,000,000 bytes: {total_bytes}")
print(f"[preflight] artifact_bytes={artifact_bytes}")
print(f"[preflight] bytes_total={total_bytes}")
print(f"[preflight] val_bpb={float(result['val']['bpb']):.8f}")
PY
