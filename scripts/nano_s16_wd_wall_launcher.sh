#!/usr/bin/env bash
# nano_s16_wd_wall_launcher.sh — ADAM_WD near-wall probe + winner replication.
#
# Chains after s15 finishes. Tests:
#   Group A: ADAM_WD in {0.067, 0.070, 0.072} (seed 42) — find peak before OOM wall
#   Group B: ADAM_WD=0.065 N=3 replication (seeds 314 + 1337)
#   Group C: LOGIT_SOFTCAP=20 at ADAM_WD=0.065 x2 seeds
#
# 7 rows × ~40 min = ~4.7 hrs
#
# Armed via:
#   nohup bash scripts/nano_s16_wd_wall_launcher.sh > /dev/null 2>&1 & disown

set -u

LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
LOG=logs/nano_s16_wd_wall_launcher.log
SWEEP=scripts/sweeps/nano_s16_wd_wall.tsv
mkdir -p logs/sweep

echo "[$(date)] nano_s16 launcher armed (pid=$$)" >> "$LOG"

# Wait for s15 launcher to finish first.
while pgrep -f 'nano_s15_wd_scan_launcher' >/dev/null 2>&1; do
    sleep 60
done
# Then wait for any residual train_gpt process.
while pgrep -f 'python.*train_gpt' >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] s15 done, host idle; launching nano_s16 sweep" >> "$LOG"
bash scripts/sweep_runner.sh "$SWEEP" >> "$LOG" 2>&1
echo "[$(date)] nano_s16 launcher DONE" >> "$LOG"

# Summary
python3 - <<'PY' >> "$LOG" 2>&1 || true
import re, statistics

results = {}
with open("logs/nano_s16_wd_wall_launcher.log") as f:
    for line in f:
        m = re.search(r'\] (nano_s16_\S+) => .*pre_quant_bpb=([\d.]+)', line)
        if m:
            results[m.group(1)] = float(m.group(2))

if not results:
    print("No results yet.")
else:
    print("\n=== nano_s16 results (pre_quant_bpb, lower=better) ===")
    ref = 1.7217  # s15 winner: ADAM_WD=0.065 s42
    print(f"  Reference (s15 ADAM_WD=0.065 s42): {ref:.4f}")
    for label in sorted(results):
        bpb = results[label]
        delta = bpb - ref
        flag = " *** BEATS REF" if delta < -0.002 else ""
        print(f"  {label:40s}  {bpb:.4f}  {delta:+.4f}{flag}")
PY
