#!/usr/bin/env bash
# spark_thread_needle_launcher.sh — narrow Spark sweep to discriminate the
# corrected-envelope 2000-step winner from the long-run floor winner.

set -u

LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd "$(dirname "$0")/.."
LOG="logs/sweep/spark_thread_needle_launcher.log"
SWEEP="scripts/sweeps/spark_thread_needle_2000_s42.tsv"
mkdir -p logs/sweep

echo "[$(date)] spark_thread_needle launcher armed (pid=$$)" >> "$LOG"

while pgrep -f 'python.*train_gpt' >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] host idle; launching spark_thread_needle sweep" >> "$LOG"
bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1
echo "[$(date)] spark_thread_needle launcher DONE" >> "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || true
import re

rows = []
pattern = re.compile(r'\] (spark_tn_(mlr\d+)_slr030_s42) => .*pre_quant_bpb=([\d.]+).*quant_bpb=([\d.]+)')

with open("logs/sweep/spark_thread_needle_launcher.log", encoding="utf-8") as handle:
    for line in handle:
        match = pattern.search(line)
        if not match:
            continue
        rows.append((match.group(2), float(match.group(3)), float(match.group(4))))

print("\n=== spark_thread_needle results ===")
print("  Goal: compare matched-envelope seed-42 outcomes for MLR 0.006 vs 0.010 vs 0.015")
print(f"{'arm':>8}  {'pre_quant':>12}  {'quant':>12}")
for arm, pre_quant, quant in sorted(rows):
    print(f"  {arm:>8}  {pre_quant:>12.8f}  {quant:>12.8f}")

if rows:
    best = min(rows, key=lambda item: item[2])
    print(f"\nBest quant_bpb: {best[0]} -> {best[2]:.8f}")
PY