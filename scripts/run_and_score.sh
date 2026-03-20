#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${1:?usage: bash scripts/run_and_score.sh <profile>}"
shift || true

bash scripts/run_remote_profile.sh "$PROFILE" "$@"

LOG_PATH="logs/${RUN_ID:-${PROFILE}}.txt"
if [ ! -f "$LOG_PATH" ]; then
  echo "log file not found: $LOG_PATH" >&2
  exit 1
fi

echo
echo "Parsed summary:"
python3 scripts/parse_run.py "$LOG_PATH"
