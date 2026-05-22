#!/usr/bin/env bash
# overnight_queue.sh — chain sweeps 02..08 after a given PID exits.
#
# Usage:
#   nohup bash scripts/overnight_queue.sh <WATCH_PID> > logs/sweep/orchestrator_overnight.log 2>&1 &
#
# Continues past ordinary run failures, but aborts the chain on a manual stop.

set +e  # explicitly NOT set -e

WATCH_PID="${1:?pid required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

run_sweep_or_stop() {
  local tsv="$1"
  bash scripts/sweep_runner.sh "$tsv"
  local rc=$?
  if [[ "$rc" == "130" || "$rc" == "143" ]]; then
    log "stop requested during ${tsv}; aborting overnight queue"
    exit "$rc"
  fi
  if [[ "$rc" != "0" ]]; then
    log "WARNING: sweep_runner non-zero for ${tsv} — continuing"
  fi
}

log "overnight queue started, waiting for PID ${WATCH_PID} to finish..."
while kill -0 "$WATCH_PID" 2>/dev/null; do sleep 15; done
log "PID ${WATCH_PID} gone — starting overnight sweeps"
sleep 10

for tsv in \
    scripts/sweeps/sweep_02_variance.tsv \
    scripts/sweeps/sweep_03_qk_fine.tsv \
    scripts/sweeps/sweep_04_warmdown.tsv \
    scripts/sweeps/sweep_05_ema.tsv \
    scripts/sweeps/sweep_06_matrix_lr.tsv \
    scripts/sweeps/sweep_07_weight_decay.tsv \
    scripts/sweeps/sweep_08_loop_timing.tsv
do
  log "starting ${tsv}"
  run_sweep_or_stop "$tsv"
  log "finished ${tsv}"
  sleep 5
done

log "overnight queue complete"
log "summary:"
bash scripts/sweep_runner.sh summary || true
