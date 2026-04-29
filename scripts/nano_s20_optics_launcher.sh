#!/usr/bin/env bash
set -euo pipefail

cd /home/ghostmini
mkdir -p logs/sweep

# Single-instance guard
exec 9>/tmp/nano_s20_optics.lock
if ! flock -n 9; then
  echo "nano_s20_optics already running"
  exit 0
fi

echo "[nano_s20] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
TIMEOUT_SECS=5400 bash scripts/sweep_runner.sh scripts/sweeps/nano_s20_optics.tsv \
  > logs/sweep/queue_nano_s20_optics.log 2>&1

echo "[nano_s20] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
