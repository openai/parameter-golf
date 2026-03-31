#!/bin/bash
set -euo pipefail
# bandit_wagon_crawler_mlp — crawler MLP leaky slope sweep
#
# Flat blocks locked at MLP_LEAKY_SLOPE=0.5. Only CRAWLER_MLP_LEAKY_SLOPE varies.
#
# BW3-00: slope=0.5  CONTROL REPIN — must match BW2-00 (1.52365 ±0.002)
#                    If it misses, stop: code change has a bug.
# BW3-01: slope=0.0  pure relu_sq — max sparsity, zero negative gradient
# BW3-02: slope=0.25 light asymmetry — retains some negative signal across loops
# BW3-03: slope=0.75 less sparse — richer negative signal for FLOW corrections
# BW3-04: slope=1.0  symmetric x² — full signal, no sparsity asymmetry
#
# Decision: beat control by ≥0.005 → gate at 2000 steps → 8×H100 if confirmed
#
# Usage:
#   bash experiments/bandit_wagon_crawler_mlp/run_ablations.sh
#   ABLATION_STEPS=2000 bash experiments/bandit_wagon_crawler_mlp/run_ablations.sh
#   NPROC_PER_NODE=8 bash experiments/bandit_wagon_crawler_mlp/run_ablations.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
ABLATION_STEPS="${ABLATION_STEPS:-500}"

RESULTS=()

run_arm() {
    local arm_id="$1"
    local label="$2"
    local slope="$3"

    echo "================================================================"
    echo "  ${arm_id} — ${label}  [${ABLATION_STEPS} steps]"
    echo "================================================================"

    env \
    MAX_WALLCLOCK_SECONDS=0 \
    ITERATIONS="${ABLATION_STEPS}" \
    WARMDOWN_ITERS=0 \
    SEED="${SEED}" \
    NPROC_PER_NODE="${NPROC}" \
    CRAWLER_MLP_LEAKY_SLOPE="${slope}" \
    bash "${SCRIPT_DIR}/run.sh" 2>&1 | tee "/tmp/${arm_id}_$(date +%H%M%S).log"

    local log
    log=$(ls /tmp/${arm_id}_*.log 2>/dev/null | tail -1)
    local bpb
    bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    local raw_bpb
    raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    RESULTS+=("${arm_id}|${label}|${slope}|${raw_bpb}|${bpb}")
    echo "  -> raw_val_bpb: ${raw_bpb}  int6_sw_bpb: ${bpb}"
    echo ""
}

run_arm BW3-00 "slope=0.5 (control repin)" 0.5
run_arm BW3-01 "slope=0.0 (pure relu_sq)"  0.0
run_arm BW3-02 "slope=0.25"                0.25
run_arm BW3-03 "slope=0.75"                0.75
run_arm BW3-04 "slope=1.0 (symmetric)"     1.0

echo "================================================================"
echo "  bandit_wagon_crawler_mlp — seed ${SEED}, ${ABLATION_STEPS} steps, warmdown=0"
echo "  Flat blocks: MLP_LEAKY_SLOPE=0.5 (locked). Only crawler slope varies."
echo "  Reference: BW2-00 (shared slope=0.5) → 1.52365"
echo "================================================================"
printf "%-8s %-25s %-8s %-14s %s\n" "ARM" "LABEL" "SLOPE" "RAW_VAL_BPB" "INT6_SW_BPB"
printf "%-8s %-25s %-8s %-14s %s\n" "---" "-----" "-----" "-----------" "-----------"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label slope raw bpb <<< "${r}"
    printf "%-8s %-25s %-8s %-14s %s\n" "${arm}" "${label}" "${slope}" "${raw}" "${bpb}"
done
echo ""
echo "  Gate: BW3-00 must be 1.521–1.526 to confirm code change is clean."
echo "  Signal: any arm must beat BW3-00 by ≥0.005 to justify promotion."
echo "================================================================"
