#!/usr/bin/env bash
# sota_replay_launcher.sh — queued launcher for the SOTA replay 2x2 sweep.
#
# Polls for idle Spark, then runs scripts/sweeps/sota_replay_v1.tsv via
# sweep_runner.sh. Safe to arm while AGX/Nano are still training: only
# polls for local train_gpt processes.
#
# Armed via:
#   nohup bash scripts/sota_replay_launcher.sh > logs/sweep/sota_replay_launcher.log 2>&1 &
#   disown

set -u

# Single-instance lock (prevent duplicate launches).
LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd "$(dirname "$0")/.."
LOG=logs/sweep/sota_replay_launcher.log
SWEEP=scripts/sweeps/sota_replay_v1.tsv
mkdir -p logs/sweep

echo "[$(date)] sota-replay launcher armed (pid=$$)" >> "$LOG"

while pgrep -f 'train_gpt_sota_decoded\.py|train_gpt_agx\.py|train_gpt_mlx\.py|records/.*/train_gpt\.py' >/dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] Spark free; launching sota_replay sweep" >> "$LOG"

bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1

echo "[$(date)] sota-replay launcher DONE" >> "$LOG"
