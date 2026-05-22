#!/usr/bin/env bash
# nano_s15_wd_scan_launcher.sh — Nano s15 weight-decay fine scan (12 rows, ~8 hr).
# ADAM_WD tight bracket {0.035–0.065}, N=3 seed replication of 0.050 winner,
# MUON_WD fine scan {0.060–0.120}, safe ADAM+MUON combo.
set -u

# Single-instance lock (prevent duplicate launches).
LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd "${NANO_ROOT:-$HOME}"
LOG=logs/nano_s15_wd_scan_launcher.log
SWEEP=scripts/sweeps/nano_s15_wd_scan.tsv
mkdir -p logs

echo "[$(date)] nano_s15_wd_scan launcher armed (pid=$$)" >> "$LOG"

# Safety: wait for any straggler python from prior sweeps.
while pgrep -f 'python.*train_gpt_agx\.py' >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] Mini idle; launching s15 wd_scan sweep" >> "$LOG"
bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1
echo "[$(date)] nano_s15_wd_scan launcher DONE" >> "$LOG"
