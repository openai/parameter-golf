#!/usr/bin/env bash
# Usage: ./await_steps.sh <experiment_dir> [n=10]
#
# Blocks until <experiment_dir>/run.log contains N training-step log lines, then
# prints them. Exits early (printing whatever's there) if the python training
# process dies before reaching N — so we don't hang the full timeout on crashes.
#
# Use this immediately after launching `run_experiment.sh` in the background, to
# inspect the trajectory before committing to wait the full ~5 minutes.

set -uo pipefail

EXP_DIR="${1:-}"
N="${2:-10}"

if [[ -z "$EXP_DIR" || ! -d "$EXP_DIR" ]]; then
  echo "Usage: $0 <experiment_dir> [n=10]" >&2
  exit 1
fi

LOG="${EXP_DIR}/run.log"

while true; do
  count=$(grep -cE '^step:[0-9]+/[0-9]+ train_loss:' "$LOG" 2>/dev/null || echo 0)
  if (( count >= N )); then break; fi
  if ! pgrep -f "python train_gpt.py" > /dev/null; then
    # Python no longer running and we don't have N steps. Print what we have
    # (which might be 0 lines if it crashed during init) and exit.
    echo "(python process exited before reaching N=${N} steps; printing ${count} available lines)" >&2
    break
  fi
  sleep 1
done

grep -E '^step:[0-9]+/[0-9]+ train_loss:' "$LOG" 2>/dev/null | head -"$N"
