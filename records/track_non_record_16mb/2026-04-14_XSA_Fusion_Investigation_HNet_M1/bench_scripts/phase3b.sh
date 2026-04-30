#!/usr/bin/env bash
# Phase 3b: segfault-tolerant rerun of the patched pilots.
# Reuses the existing logs/phase3_baseline.txt from the last run.
set -eu    # NO pipefail — segfaults at quant-time are expected
cd /workspace/parameter-golf

export DATA_DIR=/workspace/parameter-golf/data/
export ITERATIONS=200
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=20
export WARMUP_STEPS=10
export MAX_WALLCLOCK_SECONDS=0
export SEED=42

mkdir -p logs

echo "=== PHASE 3b ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

run_pilot () {
    local tag="$1" patch_qkv="$2" patch_xsa="$3"
    echo
    echo "--- run: $tag  (PATCH_QKV=$patch_qkv PATCH_XSA=$patch_xsa) ---"
    rm -f final_model.pt final_model.int6.ptz
    rm -f "logs/phase3_${tag}.txt"
    local t0=$(date +%s)
    set +e
    RUN_ID="phase3_${tag}" PATCH_QKV="$patch_qkv" PATCH_XSA="$patch_xsa" \
        python3 /workspace/phase3_run.py > "/tmp/pilot_${tag}.stdout" 2>&1
    rc=$?
    set -e
    local t1=$(date +%s)
    echo "  wall: $((t1-t0))s  exit: $rc  (non-zero expected if quant path segfaults)"
    echo "  last training lines from logs/phase3_${tag}.txt:"
    if [ -f "logs/phase3_${tag}.txt" ]; then
        grep -E '^[0-9]+/[0-9]+ train_loss' "logs/phase3_${tag}.txt" | tail -3 | sed 's/^/    /'
    else
        echo "    (log file missing)"
    fi
    echo "  last 10 stdout lines:"
    tail -10 "/tmp/pilot_${tag}.stdout" | sed 's/^/    /'
}

run_pilot qkvonly  1 0
run_pilot qkv_xsa  1 1

# --- SUMMARY ---------------------------------------------------------------
echo
echo "--- SUMMARY (step_avg from logs/phase3_*.txt) ---"
printf "%-20s %12s %14s %14s\n" "run" "step_avg(ms)" "train_loss" "tok/s"
printf "%-20s %12s %14s %14s\n" "---" "------------" "----------" "-----"
for tag in baseline qkvonly qkv_xsa; do
    logf="logs/phase3_${tag}.txt"
    if [ ! -f "$logf" ]; then
        printf "%-20s %12s\n" "phase3_${tag}" "(no log)"
        continue
    fi
    final=$(grep -E '^200/200 train_loss' "$logf" | tail -1 || true)
    if [ -z "$final" ]; then
        last_seen=$(grep -E '^[0-9]+/[0-9]+ train_loss' "$logf" | tail -1)
        printf "%-20s %-40s\n" "phase3_${tag}" "(no 200/200 — last: $last_seen)"
        continue
    fi
    time_m=$(echo "$final" | grep -oE 'train_time: [0-9.]+' | awk '{print $2}')
    loss=$(echo "$final" | grep -oE 'train_loss: [0-9.]+' | awk '{print $2}')
    toks=$(echo "$final" | grep -oE 'tok/s: [0-9]+' | awk '{print $2}')
    step_ms=$(python3 -c "print(round($time_m * 60000 / 200, 1))")
    printf "%-20s %12s %14s %14s\n" "phase3_${tag}" "$step_ms" "$loss" "$toks"
done

# Also compute steady-state step_avg from steps 100..200 (after layer loop at step 70)
echo
echo "--- steady-state step_avg (last 100 steps after layer loop) ---"
for tag in baseline qkvonly qkv_xsa; do
    logf="logs/phase3_${tag}.txt"
    if [ ! -f "$logf" ]; then continue; fi
    t100=$(grep -E '^100/200 train_loss' "$logf" | grep -oE 'train_time: [0-9.]+' | awk '{print $2}')
    t200=$(grep -E '^200/200 train_loss' "$logf" | grep -oE 'train_time: [0-9.]+' | awk '{print $2}')
    if [ -n "$t100" ] && [ -n "$t200" ]; then
        step_ms=$(python3 -c "print(round(($t200 - $t100) * 60000 / 100, 1))")
        echo "  phase3_${tag}: ${step_ms}ms/step  (from ${t100}m → ${t200}m)"
    fi
done

echo "=== PHASE 3b DONE ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
