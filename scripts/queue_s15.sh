#!/usr/bin/env bash
# queue_s15.sh — Wait for queue_s14 to finish, then run s13_locked_2000 (missed earlier).
#
# s13_locked_2000: 2000 iters, ~80 min training + ~10 min val. TIMEOUT_SECS=9000.
# This is the convergence validation run that was skipped due to shell PID issue.
#
# Usage:
#   nohup bash scripts/queue_s15.sh > logs/sweep/orchestrator_s15.log 2>&1 &
#   echo "queue_s15 PID: $!"

set +e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# Wait for s14 orchestrator
PREV_PID=1401517
log "=== queue_s15 started on $(hostname) — waiting for queue_s14 (PID $PREV_PID) ==="
while kill -0 "$PREV_PID" 2>/dev/null; do
    sleep 10
done
log "PID $PREV_PID finished"

sleep 5

# ── s13_locked_2000: 2000-step convergence validation ───────────────────────
log "Starting s13_locked_2000 (2000 iters, TIMEOUT_SECS=9000)..."
TIMEOUT_SECS=9000 FAST_SMOKE=0 bash scripts/run_experiment.sh s13_locked_2000 \
  QK_GAIN_INIT=5.5 WARMDOWN_FRAC=0.64 TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1 \
  TTT_EPOCHS=1 EMA_DECAY=0.995 LOGIT_SOFTCAP=20 ITERATIONS=2000 \
  || log "WARNING: s13_locked_2000 returned non-zero"
log "s13_locked_2000 complete"

log "=== queue_s15 finished — check logs/sweep/results.csv for convergence quant_bpb ==="
