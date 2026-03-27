#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AUTO_STOP_STEP="${AUTO_STOP_STEP:-1000}"
AUTO_STOP_MAX_VAL_BPB="${AUTO_STOP_MAX_VAL_BPB:-1.405}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-gate}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 profile [profile ...]" >&2
  exit 1
fi

for profile in "$@"; do
  export RUN_ID="${RUN_ID_PREFIX}_${profile}"
  export AUTO_STOP_STEP AUTO_STOP_MAX_VAL_BPB
  echo
  echo "=== Auto-gate $profile (step=${AUTO_STOP_STEP}, max_val_bpb=${AUTO_STOP_MAX_VAL_BPB}) ==="
  NPROC_PER_NODE="$NPROC_PER_NODE" bash scripts/run_remote_profile.sh "$profile"
  tail -n 12 "logs/${RUN_ID}.txt"
done
