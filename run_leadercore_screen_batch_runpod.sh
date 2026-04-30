#!/bin/bash
set -euo pipefail

DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
SCREEN_SECONDS="${SCREEN_SECONDS:-180}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [ "$#" -eq 0 ]; then
  set -- base warmdown800 muon099 gradclip03 matrixlr006 embedlr08
fi

cd /workspace/parameter-golf

for variant in "$@"; do
  echo "=== $(date -Is) starting $variant ==="
  DATA_ROOT_MODE="$DATA_ROOT_MODE" SCREEN_SECONDS="$SCREEN_SECONDS" NPROC_PER_NODE="$NPROC_PER_NODE" \
    bash ./launch_leadercore_screen_runpod.sh "$variant"

  record_root="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_ValidEval_TempOnly_Int8Search"
  log_path="$record_root/screen_${DATA_ROOT_MODE}_${variant}/train.log"
  pid_path="$record_root/screen_${DATA_ROOT_MODE}_${variant}/train.pid"
  pid="$(cat "$pid_path")"

  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
  done

  echo "=== $(date -Is) finished $variant ==="
  tail -n 20 "$log_path" || true
done
