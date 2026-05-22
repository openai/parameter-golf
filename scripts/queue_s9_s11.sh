#!/usr/bin/env bash
# queue_s9_s11.sh — Chain sweeps 09, 10, 11 in sequence.
#
# Sweep 09 (TTT first contact):  TIMEOUT_SECS=2400 — TTT eval adds ~10min overhead.
# Sweep 10 (clip + looping):     TIMEOUT_SECS=1800 — no TTT, standard budget.
# Sweep 11 (best combo + TTT):   TIMEOUT_SECS=2400 for TTT runs — set globally.
#
# Usage:
#   nohup bash scripts/queue_s9_s11.sh > logs/sweep/orchestrator_s9_s11.log 2>&1 &
#   echo "orchestrator PID: $!"
#
# Continues past ordinary run failures, but aborts the chain on a manual stop.

set +e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

run_sweep_or_stop() {
  local name="$1"
  shift
  "$@"
  local rc=$?
  if [[ "$rc" == "130" || "$rc" == "143" ]]; then
    log "stop requested during ${name}; aborting queue"
    exit "$rc"
  fi
  if [[ "$rc" != "0" ]]; then
    log "WARNING: ${name} returned non-zero — continuing"
  fi
}

log "=== queue_s9_s11 started on $(hostname) ==="

# ── Sweep 09: TTT first contact ──────────────────────────────────────────────
log "Starting sweep_09_ttt (7 runs, TIMEOUT_SECS=2400)..."
run_sweep_or_stop "sweep_09" env TIMEOUT_SECS=2400 bash scripts/sweep_runner.sh scripts/sweeps/sweep_09_ttt.tsv
log "sweep_09 complete"
sleep 5

# ── Sweep 10: Clip sigmas + looping probes ───────────────────────────────────
log "Starting sweep_10_probes (6 runs, TIMEOUT_SECS=1800)..."
run_sweep_or_stop "sweep_10" env TIMEOUT_SECS=1800 bash scripts/sweep_runner.sh scripts/sweeps/sweep_10_probes.tsv
log "sweep_10 complete"
sleep 5

# ── Sweep 11: Best confirmed combo + TTT stack ───────────────────────────────
log "Starting sweep_11_best_combo (4 runs, TIMEOUT_SECS=2400)..."
run_sweep_or_stop "sweep_11" env TIMEOUT_SECS=2400 bash scripts/sweep_runner.sh scripts/sweeps/sweep_11_best_combo.tsv
log "sweep_11 complete"

log "=== All sweeps done. Final summary: ==="
bash scripts/sweep_runner.sh summary || true

log "=== queue_s9_s11 finished ==="
