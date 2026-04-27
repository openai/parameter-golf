#!/bin/bash
set -euo pipefail
# bandit_wagon_choke — crawler MLP per-loop choke dimension sweep
#
# BWC-00: choke=0   CONTROL REPIN — must match BW2-00 (1.52365 ±0.002)
#                   If it misses, stop: code change has a bug.
# BWC-01: choke=32  extreme compression (= inst_dim FLOW bottleneck)
# BWC-02: choke=128 moderate compression (24× reduction from 3072)
# BWC-03: choke=256 conservative (12× reduction)
# BWC-04: choke=512 minimal choke (= model_dim, 6× reduction)
#
# Decision: beat control by ≥0.005 → gate at 2000 steps → 8×H100 if confirmed
#
# Usage:
#   bash experiments/bandit_wagon_choke/run_ablations.sh
#   ABLATION_STEPS=2000 bash experiments/bandit_wagon_choke/run_ablations.sh
#   NPROC_PER_NODE=8 bash experiments/bandit_wagon_choke/run_ablations.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
ABLATION_STEPS="${ABLATION_STEPS:-500}"

RESULTS=()

run_arm() {
    local arm_id="$1"
    local label="$2"
    local choke_dim="$3"

    echo "================================================================"
    echo "  ${arm_id} — ${label}  [${ABLATION_STEPS} steps]"
    echo "================================================================"

    env \
    MAX_WALLCLOCK_SECONDS=0 \
    ITERATIONS="${ABLATION_STEPS}" \
    WARMDOWN_ITERS=0 \
    SEED="${SEED}" \
    NPROC_PER_NODE="${NPROC}" \
    CRAWLER_MLP_CHOKE_DIM="${choke_dim}" \
    bash "${SCRIPT_DIR}/run.sh" 2>&1 | tee "/tmp/${arm_id}_$(date +%H%M%S).log"

    local log
    log=$(ls /tmp/${arm_id}_*.log 2>/dev/null | tail -1)
    local bpb
    bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    local raw_bpb
    raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    local step_avg
    step_avg=$(grep -oP 'step:[0-9]+/[0-9]+.*?step_avg:\K[0-9.]+' "${log}" 2>/dev/null | tail -1 || echo "?")
    RESULTS+=("${arm_id}|${label}|${choke_dim}|${step_avg}ms|${raw_bpb}|${bpb}")
    echo "  -> step_avg: ${step_avg}ms  raw_val_bpb: ${raw_bpb}  int6_sw_bpb: ${bpb}"
    echo ""
}

run_arm BWC-00 "choke=0 (control repin)"   0
run_arm BWC-01 "choke=32 (extreme)"       32
run_arm BWC-02 "choke=128 (moderate)"    128
run_arm BWC-03 "choke=256 (conservative)" 256
run_arm BWC-04 "choke=512 (minimal)"     512

echo "================================================================"
echo "  bandit_wagon_choke — seed ${SEED}, ${ABLATION_STEPS} steps, warmdown=0"
echo "  Flat blocks: MLP unchanged. Only crawler block uses CrawlerMLP when choke>0."
echo "  Reference: BW2-00 (choke=0, XSA=11) → 1.52365"
echo "================================================================"
printf "%-8s %-25s %-8s %-12s %-14s %s\n" "ARM" "LABEL" "CHOKE" "STEP_AVG" "RAW_VAL_BPB" "INT6_SW_BPB"
printf "%-8s %-25s %-8s %-12s %-14s %s\n" "---" "-----" "-----" "--------" "-----------" "-----------"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label choke step_avg raw bpb <<< "${r}"
    printf "%-8s %-25s %-8s %-12s %-14s %s\n" "${arm}" "${label}" "${choke}" "${step_avg}" "${raw}" "${bpb}"
done
echo ""
echo "  Gate 0: BWC-00 must be 1.521–1.526 to confirm code change is clean."
echo "  Gate 1: any arm must beat BWC-00 by ≥0.005 to justify promotion."
echo "  Watch: raw val_bpb must stay flat — all delta should be in quant gap."
echo "  Watch: step_avg for overhead — choke matmuls are cheap but measure anyway."
echo "================================================================"
