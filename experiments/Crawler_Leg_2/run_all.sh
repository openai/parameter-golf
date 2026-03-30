#!/bin/bash
set -euo pipefail
# CRAWLER_LEG_2 — Combined wins + further push
#
# Builds on Crawler_Leg_1 results:
#   CL1-07 mlp=5.0:  −0.098 BPB  (quant gap 0.287, 917 steps at 655ms)
#   CL1-01 loops=3:  −0.088 BPB  (quant gap 0.288, 884 steps at 679ms)
#
# Arm structure:
#   CL2-00  baseline (loops=4, mlp=4.0, SKIP_GPTQ=1)      — fresh reference
#   CL2-01  loops=3 + mlp=5.0 (SKIP_GPTQ=1)               — PRIMARY combined hypothesis
#   CL2-02  loops=3 + mlp=5.0 + LOOP_AWARE_GPTQ=1         — full stack (GPTQ + compile)
#           + COMPILE_FULLGRAPH=1 (SKIP_GPTQ=0)
#   CL2-03  loops=2 + mlp=5.0 (SKIP_GPTQ=1)               — push loops further
#   CL2-04  loops=3 + mlp=6.0 (SKIP_GPTQ=1)               — push MLP further
#
# Usage:
#   NPROC_PER_NODE=1 bash experiments/Crawler_Leg_2/run_all.sh
#   NPROC_PER_NODE=8 bash experiments/Crawler_Leg_2/run_all.sh

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
SUMMARY_TMP="${RESULTS_DIR}/summary_${RUN_DATE}.tmp"
SUMMARY="${RESULTS_DIR}/summary_${RUN_DATE}.txt"

echo "============================================"
echo "  CRAWLER_LEG_2 — Combined Wins + Push"
echo "  Seed: ${SEED}  GPUs: ${NPROC_PER_NODE}  Wallclock: 600s/arm"
echo "  Arms: CL2-00 through CL2-04 (5 total)  Wallclock: 350s (~4k steps on 8xH100)"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE}"
echo "============================================"
echo ""

# -------------------------------------------------------------------
# run_arm <arm_id> <label> <extra env overrides...>
# All arms share the same fixed base config (Leg 1 defaults).
# SKIP_GPTQ=1 by default — arm CL2-02 overrides to 0.
# -------------------------------------------------------------------
run_arm() {
    local arm_id="$1"
    local label="$2"
    shift 2

    local log="${RESULTS_DIR}/${arm_id}_${RUN_DATE}.log"

    echo "================================================================"
    echo "  ARM ${arm_id}  —  ${label}"
    echo "  Log: ${log}"
    echo "================================================================"

    env \
        SEED="${SEED}" \
        MAX_WALLCLOCK_SECONDS=350 \
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

    # --- Extract metrics ---
    local val_bpb steps int6_bpb quant_gap

    val_bpb=$(grep -oP 'val_bpb:\K[0-9.]+' "${log}" | tail -1 || echo "?")
    steps=$(grep -oP 'stopping_early.*step:\K[0-9]+' "${log}" | tail -1 \
          || grep -oP 'step:(\K[0-9]+)/20000 val_loss' "${log}" | tail -1 \
          || echo "?")
    int6_bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || echo "?")

    # Compute quant gap if both values present
    if [[ "${val_bpb}" != "?" && "${int6_bpb}" != "?" ]]; then
        quant_gap=$(python3 -c "print(f'+{float(\"${int6_bpb}\")-float(\"${val_bpb}\"):.3f}')" 2>/dev/null || echo "?")
    else
        quant_gap="?"
    fi

    echo "${arm_id}|${label}|${steps}|${val_bpb}|${int6_bpb}|${quant_gap}" >> "${SUMMARY_TMP}"
    echo "  -> val_bpb: ${val_bpb}  int6_sw_bpb: ${int6_bpb}  quant_gap: ${quant_gap}  steps: ${steps}"
    echo ""
}

# -------------------------------------------------------------------
# Arms
# -------------------------------------------------------------------

run_arm CL2-00 "baseline (loops=4 mlp=4.0 SKIP_GPTQ=1)"

run_arm CL2-01 "loops=3 + mlp=5.0 (SKIP_GPTQ=1)" \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=5.0

# CL2-02: Full stack — SKIP_GPTQ=0 required for LOOP_AWARE_GPTQ to function.
# Note: loop-aware GPTQ calibration takes ~850s after the 600s training wallclock.
run_arm CL2-02 "loops=3 + mlp=5.0 + LOOP_AWARE_GPTQ=1 + COMPILE (SKIP_GPTQ=0)" \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=5.0 \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=1 \
    COMPILE_FULLGRAPH=1

run_arm CL2-03 "loops=2 + mlp=5.0 (SKIP_GPTQ=1)" \
    CRAWLER_LOOPS=2 \
    CRAWLER_MLP_MULT=5.0

run_arm CL2-04 "loops=3 + mlp=6.0 (SKIP_GPTQ=1)" \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=6.0

# -------------------------------------------------------------------
# Summary table
# -------------------------------------------------------------------
BASELINE_INT6=$(grep "^CL2-00|" "${SUMMARY_TMP}" | cut -d'|' -f5)
BASELINE_VAL=$(grep "^CL2-00|" "${SUMMARY_TMP}" | cut -d'|' -f4)

echo "================================================================"
echo "  CRAWLER_LEG_2 COMPLETE — ${RUN_DATE}"
echo "  Seed: ${SEED}  Baseline int6 SW BPB: ${BASELINE_INT6}  val BPB: ${BASELINE_VAL}"
echo "================================================================"
printf "%-10s %-46s %6s %8s %10s %9s %9s\n" \
    "ARM" "LABEL" "STEPS" "VAL_BPB" "INT6_SW_BPB" "Q_GAP" "DELTA"
printf "%-10s %-46s %6s %8s %10s %9s %9s\n" \
    "---" "-----" "-----" "-------" "-----------" "-----" "-----"

while IFS='|' read -r arm label steps val_bpb int6_bpb quant_gap; do
    if [[ -n "${BASELINE_INT6}" && -n "${int6_bpb}" && "${int6_bpb}" != "?" && "${arm}" != "CL2-00" ]]; then
        delta=$(python3 -c "print(f'{float(\"${int6_bpb}\")-float(\"${BASELINE_INT6}\"):+.4f}')" 2>/dev/null || echo "?")
    else
        delta="—"
    fi
    printf "%-10s %-46s %6s %8s %10s %9s %9s\n" \
        "${arm}" "${label}" "${steps}" "${val_bpb}" "${int6_bpb}" "${quant_gap}" "${delta}"
done < "${SUMMARY_TMP}"

echo "================================================================"
echo "  Full logs: ${RESULTS_DIR}/"
echo "================================================================"

mv "${SUMMARY_TMP}" "${SUMMARY}"
