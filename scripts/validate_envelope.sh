#!/usr/bin/env bash
set -euo pipefail

CURRENT_CONFIG="${1:-config_015.env}"
NEXT_CONFIG="${2:-config_010.env}"

if [[ ! -f "$CURRENT_CONFIG" ]]; then
  echo "ERROR: Missing current config file: $CURRENT_CONFIG" >&2
  exit 1
fi

if [[ ! -f "$NEXT_CONFIG" ]]; then
  echo "ERROR: Missing next config file: $NEXT_CONFIG" >&2
  exit 1
fi

ALLOWED_DIFF_KEYS=(
  "MATRIX_LR"
  "RUN_LABEL"
  "OUTPUT_TAG"
  "RUN_ID"
  "CHECKPOINT_DIR"
  "LOG_DIR"
  "RESULTS_PATH"
  "START_TIME"
)

normalize_config() {
  local input_file="$1"

  grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$input_file" \
    | sed 's/[[:space:]]//g' \
    | sort
}

is_allowed_key() {
  local key="$1"

  for allowed in "${ALLOWED_DIFF_KEYS[@]}"; do
    if [[ "$key" == "$allowed" ]]; then
      return 0
    fi
  done

  return 1
}

tmp_current="$(mktemp)"
tmp_next="$(mktemp)"
tmp_diff="$(mktemp)"

trap 'rm -f "$tmp_current" "$tmp_next" "$tmp_diff"' EXIT

normalize_config "$CURRENT_CONFIG" > "$tmp_current"
normalize_config "$NEXT_CONFIG" > "$tmp_next"

if diff -u "$tmp_current" "$tmp_next" > "$tmp_diff"; then
  echo "PASS: Configs are identical."
  exit 0
fi

bad_diffs=0

while IFS= read -r line; do
  [[ "$line" =~ ^[-+] ]] || continue
  [[ "$line" =~ ^---|^\+\+\+ ]] && continue

  key="$(echo "$line" | sed -E 's/^[-+]//' | cut -d '=' -f 1)"

  if ! is_allowed_key "$key"; then
    echo "BLOCK: Unexpected config difference detected: $line" >&2
    bad_diffs=$((bad_diffs + 1))
  fi
done < "$tmp_diff"

echo
echo "Full diff:"
cat "$tmp_diff"

echo
if [[ "$bad_diffs" -gt 0 ]]; then
  echo "FAIL: Envelope mismatch detected. Do not launch until normalized." >&2
  exit 2
fi

echo "PASS: Only allowed launch metadata / MATRIX_LR differences detected."
exit 0
