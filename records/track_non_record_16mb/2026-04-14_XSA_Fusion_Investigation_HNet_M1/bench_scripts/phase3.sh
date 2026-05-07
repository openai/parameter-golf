#!/usr/bin/env bash
# Phase 3: 200-step training pilots on 1xH100. Compares:
#   A. baseline (no patches)
#   B. +QKV fusion only
#   C. +QKV fusion + Triton-XSA
# Step_avg from the last logged line is the headline number.
set -euo pipefail
cd /workspace/parameter-golf

export DATA_DIR=/workspace/parameter-golf/data/
export ITERATIONS=200
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=20
export WARMUP_STEPS=10
export MAX_WALLCLOCK_SECONDS=0
export SEED=42

mkdir -p logs

echo "=== PHASE 3 training pilots ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

run_pilot () {
    local tag="$1" patch_qkv="$2" patch_xsa="$3"
    echo
    echo "--- run: $tag  (PATCH_QKV=$patch_qkv PATCH_XSA=$patch_xsa) ---"
    rm -f final_model.pt final_model.int6.ptz
    RUN_ID="phase3_${tag}" \
    PATCH_QKV="$patch_qkv" \
    PATCH_XSA="$patch_xsa" \
    python3 /workspace/phase3_run.py 2>&1 | tail -60
    echo
    echo "--- step_avg from logs/phase3_${tag}.txt ---"
    grep -E '^step:[0-9]+.*step_avg' "logs/phase3_${tag}.txt" | tail -10
}

run_pilot baseline 0 0
run_pilot qkvonly  1 0
run_pilot qkv_xsa  1 1

echo
echo "--- headline step_avg (last logged step) per run ---"
for tag in baseline qkvonly qkv_xsa; do
    last=$(grep -E '^step:[0-9]+.*step_avg' "logs/phase3_${tag}.txt" | tail -1)
    echo "  phase3_${tag}: $last"
done

echo "=== PHASE 3 DONE ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
