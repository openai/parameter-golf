#!/bin/bash
set -euo pipefail
# bandit_wagon_tap — encoder tap sweep
#
# BWT-00: tap=0           CONTROL REPIN — must match BW2-00 (1.52365 ±0.002)
# BWT-01: dim=32, shared  Does any tap signal help at all?
# BWT-02: dim=32, per-loop, all   CORE HYPOTHESIS — per-loop differentiated listening
# BWT-03: dim=16, per-loop, all   Less essence
# BWT-04: dim=64, per-loop, all   More essence
# BWT-05: dim=32, per-loop, deep  Deepest encoder only — closest to crawler
# BWT-06: dim=32, per-loop, shallow  Shallowest encoder only — raw signal
#
# Usage:
#   bash experiments/bandit_wagon_tap/run_ablations.sh
#   ABLATION_STEPS=2000 bash experiments/bandit_wagon_tap/run_ablations.sh
#   NPROC_PER_NODE=8 bash experiments/bandit_wagon_tap/run_ablations.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
ABLATION_STEPS="${ABLATION_STEPS:-500}"

RESULTS=()

run_arm() {
    local arm_id="$1"
    local label="$2"
    local tap_dim="$3"
    local loop_specific="$4"
    local tap_layers="$5"

    echo "================================================================"
    echo "  ${arm_id} — ${label}  [${ABLATION_STEPS} steps]"
    echo "================================================================"

    env \
    MAX_WALLCLOCK_SECONDS=0 \
    ITERATIONS="${ABLATION_STEPS}" \
    WARMDOWN_ITERS=0 \
    SEED="${SEED}" \
    NPROC_PER_NODE="${NPROC}" \
    CRAWLER_TAP_DIM="${tap_dim}" \
    CRAWLER_TAP_LOOP_SPECIFIC="${loop_specific}" \
    CRAWLER_TAP_LAYERS="${tap_layers}" \
    bash "${SCRIPT_DIR}/run.sh" 2>&1 | tee "/tmp/${arm_id}_$(date +%H%M%S).log"

    local log
    log=$(ls /tmp/${arm_id}_*.log 2>/dev/null | tail -1)
    local bpb
    bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    local raw_bpb
    raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    local step_avg
    step_avg=$(grep -oP 'step:[0-9]+/[0-9]+.*?step_avg:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    RESULTS+=("${arm_id}|${label}|${tap_dim}|${loop_specific}|${tap_layers}|${step_avg}ms|${raw_bpb}|${bpb}")
    echo "  -> step_avg: ${step_avg}ms  raw_val_bpb: ${raw_bpb}  int6_sw_bpb: ${bpb}"
    echo ""
}

run_arm BWT-00 "control (tap=0)"                    0  1  all
run_arm BWT-01 "dim=32, shared, all"               32  0  all
run_arm BWT-02 "dim=32, per-loop, all (CORE)"      32  1  all
run_arm BWT-03 "dim=16, per-loop, all"             16  1  all
run_arm BWT-04 "dim=64, per-loop, all"             64  1  all
run_arm BWT-05 "dim=32, per-loop, deep only"       32  1  deep
run_arm BWT-06 "dim=32, per-loop, shallow only"    32  1  shallow

echo "================================================================"
echo "  bandit_wagon_tap — seed ${SEED}, ${ABLATION_STEPS} steps, warmdown=0"
echo "  4F encoder has 2 layers (0=shallow, 1=deep)"
echo "  tap_proj: shared across loops | up-projection: per-loop or shared"
echo "  Reference: BW2-00 (no tap) → 1.52365"
echo "================================================================"
printf "%-8s %-30s %-5s %-5s %-8s %-12s %-14s %s\n" \
    "ARM" "LABEL" "DIM" "LOOP" "LAYERS" "STEP_AVG" "RAW_VAL_BPB" "INT6_SW_BPB"
printf "%-8s %-30s %-5s %-5s %-8s %-12s %-14s %s\n" \
    "---" "-----" "---" "----" "------" "--------" "-----------" "-----------"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label dim loop layers step_avg raw bpb <<< "${r}"
    printf "%-8s %-30s %-5s %-5s %-8s %-12s %-14s %s\n" \
        "${arm}" "${label}" "${dim}" "${loop}" "${layers}" "${step_avg}" "${raw}" "${bpb}"
done
echo ""
echo "  Gate 0: BWT-00 must be 1.521–1.526 — validates no regressions."
echo "  Gate 1: any arm must beat BWT-00 by ≥0.005 to justify promotion."
echo "  Key comparison: BWT-01 vs BWT-02 — does per-loop differentiation add value?"
echo "  Key comparison: BWT-02 vs BWT-05/06 — which encoder layers matter?"
echo "  Watch: raw val_bpb flat across arms — all delta should be in quant gap."
echo "  Watch: step_avg — tap matmuls are cheap (dim << 512) so overhead should be small."
echo "================================================================"
