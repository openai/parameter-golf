#!/usr/bin/env bash
# queue_s14.sh — Wait for queue_s13 to finish, then run sweep_14 softcap probes.
#
# sweep_14: 4 runs, TIMEOUT_SECS=2400 each (~10-12 hrs total for s13_locked_2000 + 4 probes).
#
# Usage:
#   nohup bash scripts/queue_s14.sh > logs/sweep/orchestrator_s14.log 2>&1 &
#   echo "queue_s14 PID: $!"

set +e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

PREV_PID=1222222

log "=== queue_s14 started on $(hostname) — waiting for queue_s13 (PID $PREV_PID) ==="
if kill -0 "$PREV_PID" 2>/dev/null; then
    log "Waiting for PID $PREV_PID to finish..."
    wait "$PREV_PID" || true
    log "PID $PREV_PID finished"
else
    log "PID $PREV_PID already gone — starting immediately"
fi

sleep 5

# ── Sweep 14: LOGIT_SOFTCAP probe in locked combo ────────────────────────────
log "Starting sweep_14_softcap (4 runs, TIMEOUT_SECS=2400)..."
TIMEOUT_SECS=2400 bash scripts/sweep_runner.sh scripts/sweeps/sweep_14_softcap.tsv \
  || log "WARNING: sweep_14 returned non-zero"
log "sweep_14 complete"

log "=== queue_s14 finished — check logs/sweep/results.csv ==="
