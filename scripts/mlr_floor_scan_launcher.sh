#!/usr/bin/env bash
# mlr_floor_scan_launcher.sh — launch MATRIX_LR floor sweep on Spark.

set -u

LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd "$(dirname "$0")/.."
LOG=logs/sweep/mlr_floor_scan_launcher.log
SWEEP=scripts/sweeps/mlr_floor_scan.tsv
mkdir -p logs/sweep

echo "[$(date)] mlr_floor_scan launcher armed (pid=$$)" >> "$LOG"

while pgrep -f 'python.*train_gpt' >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] host idle; launching mlr_floor_scan sweep" >> "$LOG"
bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1
echo "[$(date)] mlr_floor_scan launcher DONE" >> "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || true
import re, statistics

lock_mean = 1.39452  # current best from mlr_low_scan at MLR=0.010 (N=2)
rows = {}
with open("logs/sweep/mlr_floor_scan_launcher.log") as f:
    for line in f:
        m = re.search(r'\] (mlr_floor_(\d+)_s(\d+)) => .*quant_bpb=([\d.]+)', line)
        if m:
            lr = int(m.group(2)) / 1000.0
            seed = f"s{m.group(3)}"
            bpb = float(m.group(4))
            rows.setdefault(lr, {})[seed] = bpb

print("\n=== mlr_floor_scan results ===")
print(f"  Reference best (mlr_low_scan): 0.010 mean={lock_mean:.5f}")
print(f"{'MLR':>8}  {'s42':>10}  {'s314':>10}  {'mean':>10}  {'vs_ref':>10}")
for lr in sorted(rows):
    seeds = rows[lr]
    vals = list(seeds.values())
    mean = statistics.mean(vals) if len(vals) > 1 else vals[0]
    vs = mean - lock_mean
    s42 = f"{seeds['s42']:.5f}" if 's42' in seeds else "-"
    s314 = f"{seeds['s314']:.5f}" if 's314' in seeds else "-"
    flag = " *** NEW BEST" if vs < 0 else ""
    print(f"  {lr:6.3f}  {s42:>10}  {s314:>10}  {mean:>10.5f}  {vs:>+10.5f}{flag}")
PY
