#!/usr/bin/env bash
# mlr_fine_scan_launcher.sh — chained launcher for MATRIX_LR fine scan.
#
# Waits for the arbiter sweep to exit, then runs the 8-row fine scan
# that fills in the LR curve around MATRIX_LR=0.042.
#
# Armed via:
#   nohup bash scripts/mlr_fine_scan_launcher.sh > logs/sweep/mlr_fine_scan_launcher.log 2>&1 &
#   disown

set -u

# Single-instance lock (prevent duplicate launches).
LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd "$(dirname "$0")/.."
LOG=logs/sweep/mlr_fine_scan_launcher.log
SWEEP=scripts/sweeps/mlr_fine_scan.tsv
mkdir -p logs/sweep

echo "[$(date)] mlr_fine_scan launcher armed (pid=$$)" >> "$LOG"

# Wait for the arbiter sweep to exit (matches TSV filename, not our cmdline).
while pgrep -f 'sweeps/arbiter_scalar_vs_matrix\.tsv' >/dev/null 2>&1; do
    sleep 60
done
# And any residual train_gpt* process on this host.
while pgrep -f 'python.*train_gpt' >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] arbiter done; launching mlr_fine_scan sweep" >> "$LOG"
bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1
echo "[$(date)] mlr_fine_scan launcher DONE" >> "$LOG"

# Summary: combine arbiter + fine_scan into a single LR curve.
python3 - <<'PY' >> "$LOG" 2>&1 || true
import csv, re
from statistics import mean, stdev
rows = [r for r in csv.DictReader(open('logs/sweep/results.csv'))
        if r['label'].startswith(('arb_A_mlr042','arb_C_mlr022','mlr_fine_'))]
by_mlr = {}
for r in rows:
    m = re.search(r'MATRIX_LR=([0-9.]+)', r.get('overrides', ''))
    if not m: continue
    try:
        bpb = float(r['pre_quant_bpb'])
    except (KeyError, ValueError):
        continue
    by_mlr.setdefault(float(m.group(1)), []).append(bpb)
print("=== MATRIX_LR curve (pre_quant_bpb) ===")
for lr in sorted(by_mlr):
    vs = by_mlr[lr]
    m = mean(vs); s = stdev(vs) if len(vs) > 1 else 0.0
    print(f"  MLR={lr:<6}  N={len(vs)}  mean={m:.5f}  std={s:.5f}  runs={['%.5f'%v for v in vs]}")
best = min(by_mlr.items(), key=lambda kv: mean(kv[1]))
print(f"\nargmin MLR = {best[0]}  mean_bpb = {mean(best[1]):.5f}")
PY
