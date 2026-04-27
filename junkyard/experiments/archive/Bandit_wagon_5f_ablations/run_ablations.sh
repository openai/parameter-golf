#!/bin/bash
set -euo pipefail
# BANDIT_WAGON_5F_ABLATIONS — 4F vs 5F direct comparison + XSA coverage sweep
#
# Addresses the gap in BW ablation: BW-00 (4F+1C) was never run as a proxy arm.
# All BW comparisons were against a full-run anchor (1.18616) — not equal compute.
#
# Arms:
#   BW2-00  4F+1C, XSA=11  THE CONTROL (missing from BW)
#   BW2-01  5F+1C, XSA=14  proportional XSA for 18-block model (73%→78% coverage)
#   (BW-03  5F+1C, XSA=11  reference, already run → 1.54404)
#
# Decision rules:
#   BW2-00 < BW-03  → 5F confirmed → gate BW2-01 winner at 2000 steps before 8×H100
#   BW2-00 > BW-03  → 4F still wins → stop, do not book 8×H100
#   BW2-01 < BW-03  → XSA=14 is better → use BW2-01 config for full run candidate
#   BW2-01 ≥ BW-03  → XSA=11 (BW-03 config) is the full run candidate
#
# Step-based stopping — same training compute on any GPU count.
#   1 GPU:  500 steps ≈ 6 min/arm, ~12 min total
#   8 GPU:  500 steps ≈ 40s/arm,   ~80s total
#
# Usage:
#   bash experiments/Bandit_wagon_5f_ablations/run_ablations.sh
#   ABLATION_STEPS=2000 bash experiments/Bandit_wagon_5f_ablations/run_ablations.sh
#   NPROC_PER_NODE=8 bash experiments/Bandit_wagon_5f_ablations/run_ablations.sh

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

# BW2-00: THE CONTROL — 4F+1C at 500 steps (the measurement gap from BW ablation)
run_arm BW2-00 "4F+1C XSA=11 (control)"      NUM_FLAT_LAYERS=4 XSA_LAST_N=11

# BW2-01: 5F+1C with proportional XSA coverage (14/18 ≈ 78% vs 11/15 = 73% baseline)
run_arm BW2-01 "5F+1C XSA=14 (proportional)" NUM_FLAT_LAYERS=5 XSA_LAST_N=14

echo "================================================================"
echo "  BANDIT_WAGON_5F ABLATIONS — seed ${SEED}, ${ABLATION_STEPS} steps, warmdown=0"
echo "  Reference BW-03 (5F+1C XSA=11, from BW ablation): 1.54404"
echo "================================================================"
printf "%-8s %-30s %s\n" "ARM" "LABEL" "INT6_SW_BPB"
printf "%-8s %-30s %s\n" "---" "-----" "-----------"
printf "%-8s %-30s %s\n" "BW-03" "5F+1C XSA=11 (BW ref)" "1.54404*"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label bpb <<< "${r}"
    printf "%-8s %-30s %s\n" "${arm}" "${label}" "${bpb}"
done
echo "  * BW-03 carried from BW ablation (seed=444, 500 steps)"
echo ""
echo "  XSA coverage reference:"
echo "    4F+1C x3 = 15 blocks. XSA_LAST_N=11 → 73%"
echo "    5F+1C x3 = 18 blocks. XSA_LAST_N=11 → 61%  (BW-03)"
echo "    5F+1C x3 = 18 blocks. XSA_LAST_N=14 → 78%  (BW2-01)"
echo "================================================================"
