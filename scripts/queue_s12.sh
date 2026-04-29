#!/usr/bin/env bash
# queue_s12.sh — Wait for queue_s9_s11 to finish, then run sweep_12.
#
# Sweep 12 (long TTT validation): TIMEOUT_SECS=3600 — full quant_bpb required.
#
# Usage:
#   nohup bash scripts/queue_s12.sh > logs/sweep/orchestrator_s12.log 2>&1 &
#   echo "queue_s12 PID: $!"

set +e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

PREV_PID=674529

log "=== queue_s12 started on $(hostname) — waiting for queue_s9_s11 (PID $PREV_PID) ==="
if kill -0 "$PREV_PID" 2>/dev/null; then
    log "Waiting for PID $PREV_PID to finish..."
    wait "$PREV_PID" || true
    log "PID $PREV_PID finished"
else
    log "PID $PREV_PID already gone — starting immediately"
fi

sleep 5

# ── Sweep 12: Long TTT validation + compound probes ──────────────────────────
log "Starting sweep_12_long_ttt (5 runs, TIMEOUT_SECS=3600)..."
TIMEOUT_SECS=3600 bash scripts/sweep_runner.sh scripts/sweeps/sweep_12_long_ttt.tsv \
  || log "WARNING: sweep_12 returned non-zero"
log "sweep_12 complete"

log "=== queue_s12 finished — check logs/sweep/results.csv for quant_bpb ==="
