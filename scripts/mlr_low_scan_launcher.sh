#!/usr/bin/env bash
# mlr_low_scan_launcher.sh — launcher for low-end MATRIX_LR probe sweep.
#
# Tests MLR in {0.018, 0.015, 0.010} × N=2 seeds to determine whether the
# optimum continues below the current lock of 0.022.
#
# Armed via:
#   nohup bash scripts/mlr_low_scan_launcher.sh > logs/sweep/mlr_low_scan_launcher.log 2>&1 &
#   disown

set -u

# Single-instance lock (prevent duplicate launches).
LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd "$(dirname "$0")/.."
LOG=logs/sweep/mlr_low_scan_launcher.log
SWEEP=scripts/sweeps/mlr_low_scan.tsv
mkdir -p logs/sweep

echo "[$(date)] mlr_low_scan launcher armed (pid=$$)" >> "$LOG"

# Wait for any running train_gpt* on this host to finish.
while pgrep -f 'python.*train_gpt' >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] host idle; launching mlr_low_scan sweep" >> "$LOG"
bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1
echo "[$(date)] mlr_low_scan launcher DONE" >> "$LOG"

# Summary: compare low-end curve to known lock at 0.022
python3 - <<'PY' >> "$LOG" 2>&1 || true
import re, statistics

lock_022 = {"s42": 1.41956, "s314": 1.42026}  # mlr_fine_scan 0.030 as reference
# (arbiter 0.022 N=3 mean = 1.4027; individual seeds not extracted here)
lock_mean = 1.4027  # arbiter N=3 mean for MLR=0.022

rows = {}
with open("logs/sweep/mlr_low_scan_launcher.log") as f:
    for line in f:
        m = re.search(r'\] (mlr_low_(\d+)_s(\d+)) => .*quant_bpb=([\d.]+)', line)
        if m:
            label, lr_str, seed, bpb = m.group(1), m.group(2), m.group(3), float(m.group(4))
            lr = int(lr_str) / 1000.0
            rows.setdefault(lr, {})[f"s{seed}"] = bpb

print("\n=== mlr_low_scan results ===")
print(f"  Reference lock: MLR=0.022 mean={lock_mean:.5f} (N=3 arbiter)")
print(f"{'MLR':>8}  {'s42':>10}  {'s314':>10}  {'mean':>10}  {'vs_lock':>10}")
for lr in sorted(rows):
    seeds = rows[lr]
    vals = list(seeds.values())
    mean = statistics.mean(vals) if len(vals) > 1 else vals[0]
    vs = mean - lock_mean
    s42  = f"{seeds['s42']:.5f}"  if 's42'  in seeds else "—"
    s314 = f"{seeds['s314']:.5f}" if 's314' in seeds else "—"
    flag = " *** BEATS LOCK" if vs < 0 else ""
    print(f"  {lr:6.3f}  {s42:>10}  {s314:>10}  {mean:>10.5f}  {vs:>+10.5f}{flag}")
print()
PY
