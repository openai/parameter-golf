#!/bin/bash
set -euo pipefail
# bandit_wagon_smear — loop smeargate on/off ablation
#
# BWS-00: CRAWLER_LOOP_SMEAR=0  control repin — must match BW2-00 (1.52365 ±0.002)
# BWS-01: CRAWLER_LOOP_SMEAR=1  loop smeargate active
#
# Usage:
#   bash experiments/bandit_wagon_smear/run_ablations.sh
#   ABLATION_STEPS=2000 bash experiments/bandit_wagon_smear/run_ablations.sh
#   NPROC_PER_NODE=8 bash experiments/bandit_wagon_smear/run_ablations.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
ABLATION_STEPS="${ABLATION_STEPS:-500}"

RESULTS=()

run_arm() {
    local arm_id="$1"
    local label="$2"
    local smear="$3"

    echo "================================================================"
    echo "  ${arm_id} — ${label}  [${ABLATION_STEPS} steps]"
    echo "================================================================"

    env \
    MAX_WALLCLOCK_SECONDS=0 \
    ITERATIONS="${ABLATION_STEPS}" \
    WARMDOWN_ITERS=0 \
    SEED="${SEED}" \
    NPROC_PER_NODE="${NPROC}" \
    CRAWLER_LOOP_SMEAR="${smear}" \
    bash "${SCRIPT_DIR}/run.sh" 2>&1 | tee "/tmp/${arm_id}_$(date +%H%M%S).log"

    local log
    log=$(ls /tmp/${arm_id}_*.log 2>/dev/null | tail -1)
    local bpb
    bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    local raw_bpb
    raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    local step_avg
    step_avg=$(grep -oP 'step:[0-9]+/[0-9]+.*?step_avg:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    RESULTS+=("${arm_id}|${label}|${smear}|${step_avg}ms|${raw_bpb}|${bpb}")
    echo "  -> step_avg: ${step_avg}ms  raw_val_bpb: ${raw_bpb}  int6_sw_bpb: ${bpb}"
    echo ""
}

run_arm BWS-00 "smear=0 (control repin)" 0
run_arm BWS-01 "smear=1 (loop smeargate)" 1

echo "================================================================"
echo "  bandit_wagon_smear — seed ${SEED}, ${ABLATION_STEPS} steps, warmdown=0"
echo "  LoopSmearGate: blends each loop output with previous loop output"
echo "  Loop 0 smears with encoder output (stable anchor)"
echo "  Reference: BW2-00 (no smear) → 1.52365"
echo "================================================================"
printf "%-8s %-28s %-6s %-12s %-14s %s\n" "ARM" "LABEL" "SMEAR" "STEP_AVG" "RAW_VAL_BPB" "INT6_SW_BPB"
printf "%-8s %-28s %-6s %-12s %-14s %s\n" "---" "-----" "-----" "--------" "-----------" "-----------"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label smear step_avg raw bpb <<< "${r}"
    printf "%-8s %-28s %-6s %-12s %-14s %s\n" "${arm}" "${label}" "${smear}" "${step_avg}" "${raw}" "${bpb}"
done
echo ""
echo "  Gate 0: BWS-00 must be 1.521–1.526 to confirm clean code change."
echo "  Gate 1: BWS-01 must beat BWS-00 by ≥0.005 to justify promotion."
echo "  Watch: step_avg — smeargate is elementwise only, should be near-zero overhead."
echo "  Watch: raw val_bpb must stay flat — all delta should be in quant gap."
echo "================================================================"
