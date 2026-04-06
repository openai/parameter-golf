#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-autoresearch_$(date +%Y%m%d_%H%M%S)}"
export ITERATIONS="${ITERATIONS:-20000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-4800}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export EVAL_STRIDE="${EVAL_STRIDE:-99999}"

bash start_sota.sh 2>&1

# Parse and print result for autoresearch to read
# IMPORTANT: The SOTA script outputs "final_int6_roundtrip" not "final_int8_zlib_roundtrip"
# We match BOTH formats so this works with either baseline or SOTA scripts
LOG="logs/${RUN_ID}.txt"
if [ -f "$LOG" ]; then
    BPB=$(grep -oP 'final_\S+_roundtrip_exact val_loss:[\d.]+ val_bpb:\K[\d.]+' "$LOG" | tail -1 || echo "FAIL")
    SIZE=$(grep -oP 'Total submission size \S+: \K\d+' "$LOG" | tail -1 || echo "0")
    echo "RESULT: val_bpb=${BPB} artifact_bytes=${SIZE}"
    if [ "$SIZE" -gt 16000000 ] 2>/dev/null; then
        echo "FAIL: Artifact size ${SIZE} exceeds 16MB limit"
    fi
else
    echo "RESULT: FAIL (no log file)"
fi
