#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-80}"
MIN_SECONDS_LEFT_TO_START="${MIN_SECONDS_LEFT_TO_START:-900}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-budget}"
AUTO_STOP_STEP="${AUTO_STOP_STEP:-1000}"
AUTO_STOP_MAX_VAL_BPB="${AUTO_STOP_MAX_VAL_BPB:-1.405}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
start_ts="$(date +%s)"
budget_seconds="$((TOTAL_BUDGET_MINUTES * 60))"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 profile [profile ...]" >&2
  exit 1
fi

for profile in "$@"; do
  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  remaining="$((budget_seconds - elapsed))"
  if [ "$remaining" -lt "$MIN_SECONDS_LEFT_TO_START" ]; then
    echo "=== Budget stop: remaining=${remaining}s is below minimum start window ${MIN_SECONDS_LEFT_TO_START}s ==="
    exit 0
  fi
  export RUN_ID="${RUN_ID_PREFIX}_${profile}"
  export AUTO_STOP_STEP AUTO_STOP_MAX_VAL_BPB
  echo
  echo "=== Budget run $profile elapsed=${elapsed}s remaining=${remaining}s ==="
  NPROC_PER_NODE="$NPROC_PER_NODE" bash scripts/run_remote_profile.sh "$profile"
  tail -n 12 "logs/${RUN_ID}.txt"
done
