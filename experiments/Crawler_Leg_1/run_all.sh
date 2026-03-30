#!/bin/bash
set -euo pipefail
# CRAWLER_LEG_1 — full ablation sequencer
# Runs all 11 arms back-to-back, prints summary table at end.
# Key metric: final val_bpb (SKIP_GPTQ=1, no quant metrics)
#
# Usage:
#   NPROC_PER_NODE=1 bash experiments/Crawler_Leg_1/run_all.sh
#   NPROC_PER_NODE=8 bash experiments/Crawler_Leg_1/run_all.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NITRUST_ENABLE="${NITRUST_ENABLE:-0}"
NITRUST_STRICT="${NITRUST_STRICT:-0}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

RUN_DATE="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${RESULTS_DIR}/summary_${RUN_DATE}.txt"

echo "============================================"
echo "  CRAWLER_LEG_1 — Full Ablation Sweep"
echo "  Seed: ${SEED}  GPUs: ${NPROC_PER_NODE}  Wallclock: 600s/arm"
echo "  Arms: CL1-00 through CL1-10 (11 total)"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE}"
echo "============================================"
echo ""

# -------------------------------------------------------------------
# run_arm <arm_id> <label> <extra env overrides...>
# -------------------------------------------------------------------
run_arm() {
    local arm_id="$1"
    local label="$2"
    shift 2
    # remaining args are KEY=VALUE overrides

    local log="${RESULTS_DIR}/${arm_id}_${RUN_DATE}.log"

    echo "================================================================"
    echo "  ARM ${arm_id}  —  ${label}"
    echo "  Log: ${log}"
    echo "================================================================"

    env \
        SEED="${SEED}" \
        MAX_WALLCLOCK_SECONDS=600 \
        WARMDOWN_ITERS=2000 \
        COMPLEMENT_ALPHA=0 \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=2048 \
        ROPE_DIMS=16 \
        SWA_EVERY=50 \
        MTP_NUM_HEADS=0 \
        LATE_QAT_THRESHOLD=0 \
        MATRIX_LR=0.03 \
        TORCHDYNAMO_OPTIMIZE_DDP=0 \
        COMPILE_FULLGRAPH=0 \
        NGRAM_EVAL_ORDER=0 \
        USE_CRAWLER=1 \
        NUM_FLAT_LAYERS=4 \
        NUM_CRAWLER_LAYERS=1 \
        CRAWLER_LOOPS=4 \
        CRAWLER_MLP_MULT=4.0 \
        INST_DIM=32 \
        CRAWLER_QUANT_INT8=1 \
        DELTA_NET_HEADS=0 \
        SKIP_EMA=1 \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0 \
        NITRUST_ENABLE="${NITRUST_ENABLE}" \
        NITRUST_STRICT="${NITRUST_STRICT}" \
        NITRUST_SO_PATH="${NITRUST_SO_PATH}" \
        "$@" \
        torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
            "${REPO_ROOT}/experiments/Medusa/train_gpt.py" \
        2>&1 | tee "${log}"

    # extract final val_bpb (last val eval line before stop)
    local val_bpb
    val_bpb=$(grep -oP 'val_bpb:\K[0-9.]+' "${log}" | tail -1)
    local steps
    steps=$(grep -oP 'stopping_early.*step:\K[0-9]+' "${log}" | tail -1 \
            || grep -oP 'step:(\K[0-9]+)/20000 val_loss' "${log}" | tail -1 \
            || echo "?")

    echo "${arm_id}|${label}|${steps}|${val_bpb}" >> "${SUMMARY}.tmp"
    echo "  -> val_bpb: ${val_bpb}  steps: ${steps}"
    echo ""
}

# -------------------------------------------------------------------
# Arms
# -------------------------------------------------------------------

run_arm CL1-00 "baseline (loops=4 inst=32 mlp=4.0 4F+1C)"

run_arm CL1-01 "loops=3" \
    CRAWLER_LOOPS=3

run_arm CL1-02 "loops=5" \
    CRAWLER_LOOPS=5

run_arm CL1-03 "inst_dim=0 (off)" \
    INST_DIM=0

run_arm CL1-04 "inst_dim=16 (narrow)" \
    INST_DIM=16

run_arm CL1-05 "inst_dim=64 (wide)" \
    INST_DIM=64

run_arm CL1-06 "mlp_mult=3.0 (narrow)" \
    CRAWLER_MLP_MULT=3.0

run_arm CL1-07 "mlp_mult=5.0 (wide)" \
    CRAWLER_MLP_MULT=5.0

run_arm CL1-08 "crawler_quant_int8=0" \
    CRAWLER_QUANT_INT8=0

run_arm CL1-09 "5F+1C" \
    NUM_FLAT_LAYERS=5 NUM_CRAWLER_LAYERS=1

run_arm CL1-10 "3F+2C" \
    NUM_FLAT_LAYERS=3 NUM_CRAWLER_LAYERS=2

# -------------------------------------------------------------------
# Summary table
# -------------------------------------------------------------------
BASELINE_BPB=$(grep "^CL1-00|" "${SUMMARY}.tmp" | cut -d'|' -f4)

echo "================================================================"
echo "  CRAWLER_LEG_1 COMPLETE — ${RUN_DATE}"
echo "  Seed: ${SEED}  Baseline val_bpb: ${BASELINE_BPB}"
echo "================================================================"
printf "%-10s %-35s %6s %8s %9s\n" "ARM" "LABEL" "STEPS" "VAL_BPB" "DELTA"
printf "%-10s %-35s %6s %8s %9s\n" "---" "-----" "-----" "-------" "-----"

while IFS='|' read -r arm label steps bpb; do
    if [[ -n "${BASELINE_BPB}" && -n "${bpb}" && "${arm}" != "CL1-00" ]]; then
        delta=$(python3 -c "print(f'{float(\"${bpb}\")-float(\"${BASELINE_BPB}\"):+.4f}')" 2>/dev/null || echo "?")
    else
        delta="—"
    fi
    printf "%-10s %-35s %6s %8s %9s\n" "${arm}" "${label}" "${steps}" "${bpb}" "${delta}"
done < "${SUMMARY}.tmp"

echo "================================================================"
echo "  Full logs: ${RESULTS_DIR}/"
echo "================================================================"

mv "${SUMMARY}.tmp" "${SUMMARY}"
