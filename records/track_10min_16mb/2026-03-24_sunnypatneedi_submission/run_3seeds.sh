#!/bin/bash
# =============================================================================
# 3-Seed Validation Run for Parameter Golf Submission
# =============================================================================
# Usage (on RunPod 8xH100):
#   cd /workspace/parameter-golf
#   nohup bash records/track_10min_16mb/2026-03-24_sunnypatneedi_submission/run_3seeds.sh > /workspace/run_3seeds.log 2>&1 &
#   tail -f /workspace/run_3seeds.log
#
# What it does:
#   1. Runs seed 42 first (sanity check — abort early if bad)
#   2. Runs seeds 1337 and 2024
#   3. Extracts metrics, computes mean/std, checks submission criteria
#   4. Copies logs into submission folder
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUB_DIR="$SCRIPT_DIR"
TRAIN_SCRIPT="$SUB_DIR/train_gpt.py"

# Env vars shared across all seeds
export XSA_LAST_N=4
export EVAL_STRIDE=64
export MAX_WALLCLOCK_SECONDS=600
# TTT config (PR #481 AdamW recipe)
export TTT_ENABLED=1
export TTT_LR=0.0005
export TTT_EPOCHS=30
export TTT_COSINE=1
export TTT_PERLAYER=1
export TTT_FREEZE_BLOCKS=0
export TTT_BATCH_SEQS=64
export TTT_MAX_STEPS=300

SEEDS=(42 1337 2024)
SOTA_BPB="1.1194"  # current merged SOTA to beat

echo "=============================================="
echo "Parameter Golf 3-Seed Validation"
echo "=============================================="
echo "Script: $TRAIN_SCRIPT"
echo "Seeds: ${SEEDS[*]}"
echo "SOTA to beat: $SOTA_BPB (need <= $(echo "$SOTA_BPB - 0.005" | bc))"
echo "Started: $(date)"
echo "=============================================="

# Check train script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: train_gpt.py not found at $TRAIN_SCRIPT"
    echo "Copy your train_gpt.py into the submission folder first!"
    exit 1
fi

# Check data exists
if [ ! -d "$REPO_DIR/data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading data..."
    python3 "$REPO_DIR/data/cached_challenge_fineweb.py" --variant sp1024
fi

# ---- Seed 42 (sanity check) ----
echo ""
echo "====== SEED 42 (sanity check) ======"
echo "Start: $(date)"
SEED=42 RUN_ID=seed42 torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT" \
    2>&1 | tee /workspace/run_seed42.log

# Extract seed 42 BPB for sanity check
BPB_42=$(grep "final_int6_sliding_window_exact" /workspace/run_seed42.log | tail -1 | grep -oP 'val_bpb:\K[\d.]+')
ARTIFACT_42=$(grep "Total submission size" /workspace/run_seed42.log | tail -1 | grep -oP '[\d]+')

echo ""
echo ">>> Seed 42 result: BPB=$BPB_42, Artifact=$ARTIFACT_42 bytes"

# Sanity check: if BPB is worse than baseline (1.2244), something is broken
if [ -n "$BPB_42" ]; then
    WORSE=$(echo "$BPB_42 > 1.20" | bc -l)
    if [ "$WORSE" -eq 1 ]; then
        echo "!!! ABORT: Seed 42 BPB ($BPB_42) is worse than 1.20 — something is broken."
        echo "!!! Fix the issue before burning 2 more seeds (~40 min, ~$16)."
        exit 1
    fi
    OVER_SIZE=$(echo "${ARTIFACT_42:-0} > 16000000" | bc)
    if [ "$OVER_SIZE" -eq 1 ]; then
        echo "!!! ABORT: Artifact ($ARTIFACT_42) exceeds 16MB limit."
        exit 1
    fi
    echo ">>> Seed 42 passed sanity check. Continuing with seeds 1337 and 2024."
else
    echo "!!! WARNING: Could not extract BPB from seed 42 log. Check manually."
    echo "!!! Continuing anyway — press Ctrl+C within 10s to abort."
    sleep 10
fi

# ---- Seed 1337 ----
echo ""
echo "====== SEED 1337 ======"
echo "Start: $(date)"
SEED=1337 RUN_ID=seed1337 torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT" \
    2>&1 | tee /workspace/run_seed1337.log

# ---- Seed 2024 ----
echo ""
echo "====== SEED 2024 ======"
echo "Start: $(date)"
SEED=2024 RUN_ID=seed2024 torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT" \
    2>&1 | tee /workspace/run_seed2024.log

# ---- Collect Results ----
echo ""
echo "=============================================="
echo "RESULTS SUMMARY"
echo "=============================================="

declare -a BPBS
declare -a ARTIFACTS
declare -a STEPS

for seed in "${SEEDS[@]}"; do
    LOG="/workspace/run_seed${seed}.log"
    BPB=$(grep "final_int6_sliding_window_exact" "$LOG" | tail -1 | grep -oP 'val_bpb:\K[\d.]+')
    ART=$(grep "Total submission size" "$LOG" | tail -1 | grep -oP '[\d]+')
    STP=$(grep "stopping_early\|step:" "$LOG" | tail -1 | grep -oP 'step[: ]*\K[\d]+' | tail -1)
    BPBS+=("$BPB")
    ARTIFACTS+=("$ART")
    STEPS+=("$STP")
    echo "Seed $seed: BPB=$BPB  Artifact=$ART bytes  Steps=$STP"
done

# Compute mean and std with python
echo ""
python3 -c "
import sys
bpbs = [float(x) for x in '${BPBS[0]},${BPBS[1]},${BPBS[2]}'.split(',') if x]
if len(bpbs) == 3:
    mean = sum(bpbs) / len(bpbs)
    std = (sum((x - mean)**2 for x in bpbs) / len(bpbs))**0.5
    sota = $SOTA_BPB
    delta = mean - sota
    print(f'Mean BPB: {mean:.4f} (std {std:.4f})')
    print(f'vs SOTA ({sota}): {delta:+.4f} nats')
    if delta < -0.005:
        print(f'PASS: Beats SOTA by {abs(delta):.4f} nats (>= 0.005 required)')
    elif delta < 0:
        print(f'CLOSE: Improves by {abs(delta):.4f} nats but < 0.005 threshold')
    else:
        print(f'FAIL: Does not beat SOTA')

    # Check all artifacts under 16MB
    arts = [int(x) for x in '${ARTIFACTS[0]},${ARTIFACTS[1]},${ARTIFACTS[2]}'.split(',') if x]
    all_under = all(a < 16_000_000 for a in arts)
    print(f'Artifacts under 16MB: {\"PASS\" if all_under else \"FAIL\"} (max: {max(arts)} bytes)')
else:
    print(f'ERROR: Only got {len(bpbs)} BPB values, expected 3')
"

# Copy logs into submission folder
echo ""
echo "Copying logs to submission folder..."
for seed in "${SEEDS[@]}"; do
    cp "/workspace/run_seed${seed}.log" "$SUB_DIR/train_seed${seed}.log"
    echo "  Copied train_seed${seed}.log"
done

echo ""
echo "=============================================="
echo "NEXT STEPS (run locally after scp):"
echo "=============================================="
echo "1. scp -r runpod:/workspace/parameter-golf/records/track_10min_16mb/2026-03-24_sunnypatneedi_submission/ ."
echo "2. Update submission.json with actual val_bpb and bytes_total"
echo "3. Update README.md results table"
echo "4. Rename folder if desired"
echo "5. git checkout -b submission/sunnypatneedi-leakyrelu-xsa"
echo "6. git add records/track_10min_16mb/2026-03-24_sunnypatneedi_submission/"
echo "7. git commit"
echo "8. git push origin submission/sunnypatneedi-leakyrelu-xsa"
echo "9. Open PR at: https://github.com/openai/parameter-golf/compare"
echo ""
echo "Finished: $(date)"
echo "YOU CAN STOP THE RUNPOD POD NOW."
