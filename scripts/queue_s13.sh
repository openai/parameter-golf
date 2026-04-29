#!/usr/bin/env bash
# queue_s13.sh — Run sweep_13 long validation (locked config at 500 + 2000 steps).
#
# s13_locked_500:  ~30 min training + ~10 min val = ~40 min. TIMEOUT_SECS=3600.
# s13_locked_2000: ~68 min training + ~10 min val = ~80 min. TIMEOUT_SECS=9000.
#
# Usage:
#   nohup bash scripts/queue_s13.sh > logs/sweep/orchestrator_s13.log 2>&1 &
#   echo "queue_s13 PID: $!"

set +e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "=== queue_s13 started on $(hostname) ==="

# ── Sweep 13: Full-scale locked-config validation ────────────────────────────
log "Starting sweep_13_long_validate (2 runs — 500 steps then 2000 steps)..."
TIMEOUT_SECS=9000 bash scripts/sweep_runner.sh scripts/sweeps/sweep_13_long_validate.tsv \
  || log "WARNING: sweep_13 returned non-zero"
log "sweep_13 complete"

log "=== queue_s13 finished — check logs/sweep/results.csv for quant_bpb ==="
