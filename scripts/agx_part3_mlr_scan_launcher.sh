#!/usr/bin/env bash
# agx_part3_mlr_scan_launcher.sh — AGX MATRIX_LR fine-scan (5 rows, ~10 hr).
# Cross-validates historical peak MLR=0.040 vs today's s42 winner at 0.042,
# plus N=2 seed pair at 0.042 and brackets 0.038 / 0.044.
set -u

# Single-instance lock (prevent duplicate launches).
LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd /mnt/nvme
LOG=logs/sweep/agx_part3_mlr_scan_launcher.log
SWEEP=scripts/sweeps/agx_part3_mlr_scan.tsv
mkdir -p logs/sweep

echo "[$(date)] agx_part3_mlr_scan launcher armed (pid=$$)" >> "$LOG"

# Safety: wait for any straggler python from prior sweeps.
while pgrep -f 'python.*train_gpt_agx\.py' >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] AGX idle; launching part3 mlr_scan sweep" >> "$LOG"
bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1
echo "[$(date)] agx_part3_mlr_scan launcher DONE" >> "$LOG"
