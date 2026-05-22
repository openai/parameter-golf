#!/usr/bin/env bash
# arbiter_launcher.sh — queued launcher for scalar-vs-matrix LR arbiter sweep.
#
# Waits for the sota_replay sweep to exit, then runs the 9-row arbiter.
# Pattern matches self-safe (uses script filename, not process cmdline).
#
# Armed via:
#   nohup bash scripts/arbiter_launcher.sh > logs/sweep/arbiter_launcher.log 2>&1 &
#   disown

set -u

# Single-instance lock (prevent duplicate launches).
LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd "$(dirname "$0")/.."
LOG=logs/sweep/arbiter_launcher.log
SWEEP=scripts/sweeps/arbiter_scalar_vs_matrix.tsv
mkdir -p logs/sweep

echo "[$(date)] arbiter launcher armed (pid=$$)" >> "$LOG"

# Wait for SOTA replay to finish. Match by TSV filename (not by our own cmdline).
while pgrep -f 'sweeps/sota_replay_v1\.tsv' >/dev/null 2>&1; do
    sleep 60
done
# Also wait for any active training (belt & suspenders).
while pgrep -f 'python.*train_gpt_sota_decoded\.py|records/.*train_gpt\.py' >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] prior sweep done; launching arbiter sweep" >> "$LOG"
bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1
echo "[$(date)] arbiter launcher DONE" >> "$LOG"

# Summary: group by arm prefix and dump mean/std from results.csv.
python3 - <<'PY' >> "$LOG" 2>&1 || true
import csv, statistics as s
rows=[r for r in csv.DictReader(open('logs/sweep/results.csv')) if r['label'].startswith('arb_')]
by_arm={}
for r in rows:
    arm=r['label'].split('_')[1]  # A, B, C
    try: v=float(r['quant_bpb'])
    except: continue
    by_arm.setdefault(arm,[]).append(v)
print("=== ARBITER RESULTS ===")
for arm,vals in sorted(by_arm.items()):
    if vals:
        m=s.mean(vals); sd=s.stdev(vals) if len(vals)>1 else 0.0
        print(f"  Arm {arm}: n={len(vals)} mean={m:.5f} std={sd:.5f} vals={vals}")
PY
