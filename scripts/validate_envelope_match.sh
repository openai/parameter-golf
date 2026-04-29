#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/validate_envelope_match.sh \
    --reference-label <label_in_results_csv> \
    --candidate-overrides "KEY=VAL KEY=VAL ..." \
    [--csv logs/sweep/results.csv] \
    [--must-match "K1 K2 K3 ..."]

Purpose:
  Enforce head-to-head envelope equality. The candidate is allowed to differ only
  in knobs you intentionally omit from --must-match (for example MATRIX_LR).

Example:
  scripts/validate_envelope_match.sh \
    --reference-label sweep_hgv_confirm_mlr015_slr030_s42 \
    --candidate-overrides "QK_GAIN_INIT=5.5 WARMDOWN_FRAC=0.64 TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1 TTT_EPOCHS=1 EMA_DECAY=0.995 LOGIT_SOFTCAP=20 MATRIX_LR=0.010 SCALAR_LR=0.030 ITERATIONS=2000 SEED=42 MAX_WALLCLOCK_SECONDS=2400 TIMEOUT_SECS=5400 FAST_SMOKE=0" \
    --csv logs/sweep/results.csv
EOF
}

CSV_PATH="logs/sweep/results.csv"
REF_LABEL=""
CANDIDATE_OVERRIDES=""
MUST_MATCH="SCALAR_LR WARMDOWN_FRAC QK_GAIN_INIT TTT_ENABLED SLIDING_WINDOW_ENABLED TTT_EPOCHS EMA_DECAY LOGIT_SOFTCAP ITERATIONS SEED MAX_WALLCLOCK_SECONDS TIMEOUT_SECS FAST_SMOKE"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reference-label)
      REF_LABEL="$2"
      shift 2
      ;;
    --candidate-overrides)
      CANDIDATE_OVERRIDES="$2"
      shift 2
      ;;
    --csv)
      CSV_PATH="$2"
      shift 2
      ;;
    --must-match)
      MUST_MATCH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$REF_LABEL" || -z "$CANDIDATE_OVERRIDES" ]]; then
  usage
  exit 2
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: CSV not found: $CSV_PATH" >&2
  exit 2
fi

ref_line="$(grep -F "${REF_LABEL}," "$CSV_PATH" | tail -n 1 || true)"
if [[ -z "$ref_line" ]]; then
  echo "ERROR: No row found for reference label: $REF_LABEL" >&2
  exit 2
fi

extract_pairs() {
  local text="$1"
  # Parse uppercase env-style KEY=VALUE tokens. Values stop at whitespace/comma.
  grep -oE '[A-Z0-9_]+=[^[:space:],]+' <<<"$text" || true
}

declare -A REF
while IFS='=' read -r k v; do
  [[ -n "$k" ]] || continue
  REF["$k"]="$v"
done < <(extract_pairs "$ref_line")

declare -A CAND
while IFS='=' read -r k v; do
  [[ -n "$k" ]] || continue
  CAND["$k"]="$v"
done < <(extract_pairs "$CANDIDATE_OVERRIDES")

echo "Reference label: $REF_LABEL"
echo "CSV path: $CSV_PATH"
echo

echo "Reference envelope tokens:"
extract_pairs "$ref_line" | tr '\n' ' '
echo
echo

echo "Candidate override tokens:"
extract_pairs "$CANDIDATE_OVERRIDES" | tr '\n' ' '
echo
echo

mismatch=0
for key in $MUST_MATCH; do
  ref_v="${REF[$key]:-}"
  cand_v="${CAND[$key]:-}"
  if [[ -z "$ref_v" ]]; then
    echo "[WARN] Reference missing key: $key"
    mismatch=1
    continue
  fi
  if [[ -z "$cand_v" ]]; then
    echo "[FAIL] Candidate missing key: $key (reference has $ref_v)"
    mismatch=1
    continue
  fi
  if [[ "$ref_v" != "$cand_v" ]]; then
    echo "[FAIL] $key mismatch: reference=$ref_v candidate=$cand_v"
    mismatch=1
  else
    echo "[OK]   $key=$cand_v"
  fi
done

echo
if [[ $mismatch -ne 0 ]]; then
  echo "Envelope check: FAIL"
  exit 1
fi

echo "Envelope check: PASS"
echo "Allowed differences are any keys excluded from --must-match (for example MATRIX_LR)."
