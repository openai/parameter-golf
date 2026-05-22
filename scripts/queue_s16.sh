#!/usr/bin/env bash
# queue_s16.sh — Run sweep_16 (MATRIX_LR probe) after s13_locked_2000 finishes.
#
# Waits for s13_locked_2000 relaunch (PID 1618040), then runs 7 matrix_lr probes.
# All 7 short runs ~7×25min = ~3hr. TTT enabled requires TIMEOUT_SECS=2400.
#
# Usage:
#   nohup bash scripts/queue_s16.sh > logs/sweep/orchestrator_s16.log 2>&1 &
#   echo "queue_s16 PID: $!"

set +e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# Wait for s13_locked_2000 relaunch
PREV_PID=1618040
log "=== queue_s16 started on $(hostname) — waiting for s13_locked_2000 (PID $PREV_PID) ==="
while kill -0 "$PREV_PID" 2>/dev/null; do
    sleep 30
done
log "PID $PREV_PID finished"

sleep 5

log "=== Starting sweep_16 — MATRIX_LR probe (7 runs, ~3hr) ==="
TIMEOUT_SECS=2400 bash scripts/sweep_runner.sh scripts/sweeps/sweep_16_matrix_lr.tsv
log "=== sweep_16 complete — check logs/sweep/results.csv ==="
