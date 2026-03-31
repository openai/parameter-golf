#!/bin/bash
set -euo pipefail
# bandit_wagon_XSA — XSA coverage sweep on confirmed-optimal 4F+1C config
#
# Hypothesis: wider XSA smooths quantization perturbation by providing cross-block
# bandwidth. Raw learning rate is unaffected; gain is purely quant robustness.
#
# 4F+1C x3 = 15 total blocks. Coverage:
#   XSA_LAST_N=11 → 73%  (control — BW2-00: 1.52365, 546ms/step)
#   XSA_LAST_N=13 → 87%  (BWXSA-01)
#   XSA_LAST_N=15 → 100% (BWXSA-02 — ceiling)
#
# Decision rule:
#   improvement AND step overhead <8% (+44ms vs 546ms baseline) → gate at 2000 steps
#   no improvement at XSA=15                                     → XSA=11 is optimal, stop
#
# IMPORTANT: record step_avg from each arm — that is the speed signal.
#
# Usage:
#   bash experiments/bandit_wagon_XSA/run_ablations.sh
#   ABLATION_STEPS=2000 bash experiments/bandit_wagon_XSA/run_ablations.sh
#   NPROC_PER_NODE=8 bash experiments/bandit_wagon_XSA/run_ablations.sh

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
    local step_avg
    step_avg=$(grep -oP 'step:[0-9]+/[0-9]+ train_loss:[0-9.]+ train_time:[0-9]+ms step_avg:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    RESULTS+=("${arm_id}|${label}|${step_avg}|${bpb}")
    echo "  -> int6_sw_bpb: ${bpb}  step_avg: ${step_avg}ms"
    echo ""
}

run_arm BWXSA-01 "4F+1C XSA=13 (87%)"  XSA_LAST_N=13
run_arm BWXSA-02 "4F+1C XSA=15 (100%)" XSA_LAST_N=15

echo "================================================================"
echo "  bandit_wagon_XSA ABLATIONS — seed ${SEED}, ${ABLATION_STEPS} steps, warmdown=0"
echo "  Control: BW2-00 (4F+1C XSA=11, 73%) → 1.52365, 546ms/step"
echo "================================================================"
printf "%-10s %-25s %-14s %s\n" "ARM" "LABEL" "STEP_AVG(ms)" "INT6_SW_BPB"
printf "%-10s %-25s %-14s %s\n" "---" "-----" "------------" "-----------"
printf "%-10s %-25s %-14s %s\n" "Control" "4F+1C XSA=11 (73%)" "546ms*" "1.52365*"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label step_avg bpb <<< "${r}"
    printf "%-10s %-25s %-14s %s\n" "${arm}" "${label}" "${step_avg}ms" "${bpb}"
done
echo "  * control from BW5F ablation (same seed, same steps, same config)"
echo ""
echo "  Overhead threshold: <8% step increase (~+44ms over 546ms) to net positive at 600s"
echo "================================================================"
