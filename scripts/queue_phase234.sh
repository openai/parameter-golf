#!/usr/bin/env bash
# queue_phase234.sh — waits for phase 1 orchestrator to finish, then runs
# phases 2, 3, 4 sequentially. Designed to be nohup'd in the background.
#
# Usage:
#   nohup bash scripts/queue_phase234.sh <phase1_orchestrator_pid> \
#     > logs/sweep/orchestrator_phase234.log 2>&1 &
#
# If no PID given, scans pgrep for any active sweep_runner/run_experiment.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WAIT_PID="${1:-}"

run_sweep_or_stop() {
  local tsv="$1"
  bash scripts/sweep_runner.sh "$tsv"
  local rc=$?
  if [[ "$rc" == "130" || "$rc" == "143" ]]; then
    echo "[queue] stop requested during $tsv; aborting queue"
    exit "$rc"
  fi
  if [[ "$rc" != "0" ]]; then
    echo "[queue] $tsv exited non-zero (continuing)"
  fi
}

echo "[queue] started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ -n "$WAIT_PID" ]]; then
  echo "[queue] waiting for PID $WAIT_PID to exit..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[queue] PID $WAIT_PID exited at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
  # Fallback: wait for any active sweep process to clear
  echo "[queue] no PID given; polling for active sweep processes..."
  while pgrep -f "sweep_runner.sh|run_experiment.sh|train_gpt_sota_decoded" >/dev/null 2>&1; do
    sleep 30
  done
  echo "[queue] no active sweep processes at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

# Grace delay so GPU memory settles before Phase 2 starts
sleep 10

for tsv in \
  scripts/sweeps/phase2_sanity.tsv \
  scripts/sweeps/phase3_probes.tsv \
  scripts/sweeps/phase4_novel.tsv
do
  if [[ ! -f "$tsv" ]]; then
    echo "[queue] MISSING $tsv — skipping"
    continue
  fi
  echo ""
  echo "############################################################"
  echo "# [queue] launching $tsv at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "############################################################"
  run_sweep_or_stop "$tsv"
done

echo ""
echo "[queue] all phases complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "--- final results.csv ---"
cat logs/sweep/results.csv
