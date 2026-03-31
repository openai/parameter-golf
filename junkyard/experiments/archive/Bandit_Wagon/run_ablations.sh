#!/bin/bash
set -euo pipefail
# BANDIT_WAGON — width/depth ablation sweep
# Runs BW-01 through BW-04 back to back, step-capped, warmdown off.
# BW-00 anchor: 1.18616 (seed 444, 600s, 8×H100)
#
# Step-based stopping — same training compute on any GPU count.
#   1 GPU:  500 steps ≈ 6 min/arm, 24 min total
#   8 GPU:  500 steps ≈ 40s/arm,   3 min total
#
# Usage:
#   bash experiments/Bandit_Wagon/run_ablations.sh              # 1 GPU, 500 steps
#   ABLATION_STEPS=1000 bash experiments/Bandit_Wagon/run_ablations.sh
#   NPROC_PER_NODE=8 bash experiments/Bandit_Wagon/run_ablations.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
ABLATION_STEPS="${ABLATION_STEPS:-500}"

RESULTS=()

run_arm() {
    local arm_id="$1"
    local label="$2"
    shift 2

    echo "================================================================"
    echo "  ${arm_id} — ${label}  [${ABLATION_STEPS} steps]"
    echo "================================================================"

    env \
    MAX_WALLCLOCK_SECONDS=0 \
    ITERATIONS="${ABLATION_STEPS}" \
    WARMDOWN_ITERS=0 \
    SEED="${SEED}" \
    NPROC_PER_NODE="${NPROC}" \
    "$@" \
    bash "${SCRIPT_DIR}/run.sh" 2>&1 | tee "/tmp/${arm_id}_$(date +%H%M%S).log"

    local log
    log=$(ls /tmp/${arm_id}_*.log 2>/dev/null | tail -1)
    local bpb
    bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    RESULTS+=("${arm_id}|${label}|${bpb}")
    echo "  -> int6_sw_bpb: ${bpb}"
    echo ""
}

run_arm BW-01 "dim=576 (width narrow)"  MODEL_DIM=576
run_arm BW-02 "dim=640 (width wide)"    MODEL_DIM=640
run_arm BW-03 "5F+1C (depth +1)"        NUM_FLAT_LAYERS=5
run_arm BW-04 "6F+1C (depth +2)"        NUM_FLAT_LAYERS=6

echo "================================================================"
echo "  BANDIT_WAGON ABLATIONS — seed ${SEED}, ${ABLATION_STEPS} steps, warmdown=0"
echo "  Anchor BW-00 (dim=512, 4F): 1.18616 (8000 steps, 600s, 8xH100)"
echo "================================================================"
printf "%-8s %-25s %s\n" "ARM" "LABEL" "INT6_SW_BPB"
printf "%-8s %-25s %s\n" "---" "-----" "-----------"
printf "%-8s %-25s %s\n" "BW-00" "dim=512 4F (anchor)" "1.18616*"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label bpb <<< "${r}"
    printf "%-8s %-25s %s\n" "${arm}" "${label}" "${bpb}"
done
echo "  * anchor at full 600s — proxy arms are directional only"
echo "================================================================"
