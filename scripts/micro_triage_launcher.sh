#!/usr/bin/env bash
# micro_triage_launcher.sh — waits for current Spark training to finish,
# then runs Tier A GPTQ micro-tests against the two best s34 checkpoints.
# Also triggers triage_score at the end.

set -u

# Single-instance lock (prevent duplicate launches).
LOCKFILE="/tmp/$(basename "$0" .sh).lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "[$(date)] $(basename "$0") already running (lock=$LOCKFILE); exiting pid=$$" >&2; exit 0; }

cd "$(dirname "$0")/.."
LOG=logs/sweep/micro_triage_launcher.log
mkdir -p logs/sweep records/ckpts

echo "[$(date)] launcher armed" >> "$LOG"

# Wait for any running train_gpt_sota_decoded to exit (s34_seed314 currently)
while pgrep -f 'train_gpt_sota_decoded\.py' >/dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] Spark training free; starting Tier A" >> "$LOG"

# Run Tier A on two seed checkpoints (N=2 → cross-validate knob directionality)
for ckpt_tag in "s34_soup_seed42" "s34_soup_seed1337"; do
    CKPT="records/ckpts/${ckpt_tag}.pt"
    if [[ ! -f "$CKPT" ]]; then
        echo "[$(date)] SKIP $CKPT (missing)" >> "$LOG"
        continue
    fi
    echo "[$(date)] ==== Tier A: $ckpt_tag ====" >> "$LOG"
    python3 -u scripts/micro_gptq_triage.py --ckpt "$CKPT" --tag "$ckpt_tag" >> "$LOG" 2>&1
done

echo "[$(date)] Tier A complete; scoring..." >> "$LOG"
python3 scripts/triage_score.py --tier a >> "$LOG" 2>&1
echo "[$(date)] Spark Tier A DONE — awaiting user decision on 3000-step confirmation" >> "$LOG"
# NOTE: Tier B moved to AGX (runs in parallel, longer 2000-step runs).
# Spark remains free after Tier A for the 3000-step confirmation run (user-gated).
