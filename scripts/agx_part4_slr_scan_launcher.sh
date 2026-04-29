#!/usr/bin/env bash
# agx_part4_slr_scan_launcher.sh — AGX SCALAR_LR=0.030 probe + bracket pairs (4 rows, ~8 hr).
# Waits for part3 to finish before starting.
set -u

LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd /mnt/nvme
LOG=logs/sweep/agx_part4_slr_scan_launcher.log
SWEEP=scripts/sweeps/agx_part4_slr_scan.tsv
mkdir -p logs/sweep

echo "[$(date)] agx_part4_slr_scan launcher armed (pid=$$)" >> "$LOG"

while pgrep -f 'python.*train_gpt_agx\.py' >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] AGX idle; launching part4 slr_scan sweep" >> "$LOG"
bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1
echo "[$(date)] agx_part4_slr_scan launcher DONE" >> "$LOG"
