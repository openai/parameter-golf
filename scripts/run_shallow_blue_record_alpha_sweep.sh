#!/usr/bin/env bash
set -euo pipefail

RUN_ID_BASE="${RUN_ID_BASE:-shallow_blue_record_alpha_sweep_$(date +%Y%m%d_%H%M%S)}"
ALPHAS_CSV="${ALPHAS:-0.20,0.25,0.30}"
SUBMISSION_DIR="${SUBMISSION_DIR:-records/track_10min_16mb/2026-04-07_Shallow_Blue_Probe_BOS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODEL_PATH="${MODEL_PATH:-${SUBMISSION_DIR}/final_model.int8.ptz}"
PROBE_ARTIFACT="${PROBE_ARTIFACT:-${SUBMISSION_DIR}/shallow_blue_probe.json}"
VAL_FILES="${VAL_FILES:-data/datasets/fineweb10B_sp1024/fineweb_val_*.bin}"
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizers/fineweb_1024_bpe.model}"
WINDOW="${WINDOW:-1024}"
STRIDE="${STRIDE:-1024}"
BATCH_WINDOWS="${BATCH_WINDOWS:-32}"
MAX_DOCS="${MAX_DOCS:-0}"
MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-0}"
PYTHON_BIN="${PYTHON:-python3}"
LOG_DIR="${LOG_DIR:-reports/shallow_blue}"
DRY_RUN="${DRY_RUN:-0}"

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
mkdir -p "${LOG_DIR}"

echo "Shallow Blue artifact alpha sweep"
echo "  run_id_base:   ${RUN_ID_BASE}"
echo "  submission_dir:${SUBMISSION_DIR}"
echo "  alphas:        ${ALPHAS_CSV}"
echo "  nproc:         ${NPROC_PER_NODE}"
echo "  model:         ${MODEL_PATH}"
echo "  probe:         ${PROBE_ARTIFACT}"
echo "  val files:     ${VAL_FILES}"
echo "  tokenizer:     ${TOKENIZER_PATH}"
echo "  window/stride: ${WINDOW}/${STRIDE}"
echo "  batch windows: ${BATCH_WINDOWS}"
echo "  max docs:      ${MAX_DOCS}"
echo "  max tokens:    ${MAX_VAL_TOKENS}"
echo "  log dir:       ${LOG_DIR}"
echo "  dry_run:       ${DRY_RUN}"

IFS=',' read -r -a ALPHAS_LIST <<< "${ALPHAS_CSV}"
LOGS=()

for raw_alpha in "${ALPHAS_LIST[@]}"; do
  alpha="$(printf '%s' "${raw_alpha}" | xargs)"
  if [[ -z "${alpha}" ]]; then
    continue
  fi
  alpha_tag="${alpha//./p}"
  run_id="${RUN_ID_BASE}_a${alpha_tag}"
  log_path="${LOG_DIR}/${run_id}.log"
  LOGS+=("${log_path}")

  echo
  echo "=== alpha ${alpha} ==="
  printf 'env RUN_ID=%q LOG_PATH=%q SUBMISSION_DIR=%q MODEL_PATH=%q PROBE_ARTIFACT=%q NPROC_PER_NODE=%q VAL_FILES=%q TOKENIZER_PATH=%q WINDOW=%q STRIDE=%q BATCH_WINDOWS=%q ALPHA=%q MAX_DOCS=%q MAX_VAL_TOKENS=%q PYTHON=%q bash scripts/run_shallow_blue_record_artifact_eval.sh\n' \
    "${run_id}" \
    "${log_path}" \
    "${SUBMISSION_DIR}" \
    "${MODEL_PATH}" \
    "${PROBE_ARTIFACT}" \
    "${NPROC_PER_NODE}" \
    "${VAL_FILES}" \
    "${TOKENIZER_PATH}" \
    "${WINDOW}" \
    "${STRIDE}" \
    "${BATCH_WINDOWS}" \
    "${alpha}" \
    "${MAX_DOCS}" \
    "${MAX_VAL_TOKENS}" \
    "${PYTHON_BIN}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  env \
    RUN_ID="${run_id}" \
    LOG_PATH="${log_path}" \
    SUBMISSION_DIR="${SUBMISSION_DIR}" \
    MODEL_PATH="${MODEL_PATH}" \
    PROBE_ARTIFACT="${PROBE_ARTIFACT}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    VAL_FILES="${VAL_FILES}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    WINDOW="${WINDOW}" \
    STRIDE="${STRIDE}" \
    BATCH_WINDOWS="${BATCH_WINDOWS}" \
    ALPHA="${alpha}" \
    MAX_DOCS="${MAX_DOCS}" \
    MAX_VAL_TOKENS="${MAX_VAL_TOKENS}" \
    PYTHON="${PYTHON_BIN}" \
    bash scripts/run_shallow_blue_record_artifact_eval.sh
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

echo
echo "=== sweep summary ==="
"${PYTHON_BIN}" - "${LOGS[@]}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

probe_pattern = re.compile(
    r"final_shallow_blue_probe delta_bpb:([+-]?[0-9.]+) "
    r"mixed_bpb:([0-9.]+).*mean_alpha:([0-9.]+).*boosted_rows:(\d+)"
)
exact_pattern = re.compile(
    r"final_shallow_blue_probe_exact val_bpb:([0-9.]+) "
    r"delta_bpb:([+-]?[0-9.]+) elapsed_seconds:([0-9.]+)"
)

rows = []
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    text = path.read_text(encoding="utf-8")
    alpha_match = re.search(r"_a(\d+)p(\d+)\.log$", path.name)
    if alpha_match is None:
        alpha_label = path.stem
    else:
        alpha_label = f"{alpha_match.group(1)}.{alpha_match.group(2)}"
    exact = exact_pattern.search(text)
    probe = probe_pattern.search(text)
    if exact is None:
        raise SystemExit(f"missing final_shallow_blue_probe_exact in {path}")
    val_bpb = float(exact.group(1))
    delta_bpb = float(exact.group(2))
    elapsed_seconds = float(exact.group(3))
    mean_alpha = float(probe.group(3)) if probe else float("nan")
    boosted_rows = int(probe.group(4)) if probe else -1
    rows.append(
        {
            "alpha": alpha_label,
            "val_bpb": val_bpb,
            "delta_bpb": delta_bpb,
            "elapsed_seconds": elapsed_seconds,
            "mean_alpha": mean_alpha,
            "boosted_rows": boosted_rows,
            "log_path": str(path),
        }
    )

rows.sort(key=lambda item: item["val_bpb"])
best = rows[0]
print(
    f"best_alpha:{best['alpha']} best_val_bpb:{best['val_bpb']:.8f} "
    f"delta_bpb:{best['delta_bpb']:+.8f} elapsed_seconds:{best['elapsed_seconds']:.3f}"
)
for row in rows:
    print(
        f"alpha:{row['alpha']} val_bpb:{row['val_bpb']:.8f} "
        f"delta_bpb:{row['delta_bpb']:+.8f} elapsed_seconds:{row['elapsed_seconds']:.3f} "
        f"mean_alpha:{row['mean_alpha']:.6f} boosted_rows:{row['boosted_rows']} "
        f"log:{row['log_path']}"
    )
PY
