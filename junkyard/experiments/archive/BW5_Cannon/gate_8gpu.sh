#!/bin/bash
set -euo pipefail
# ================================================================
#  Bandit_Wagon_V_Cannon — 8×H100 SPEED GATE
#
#  Purpose: verify that cannon's -20ms/step speedup (seen on 1 GPU)
#  survives DDP all-reduce on 8×H100.
#
#  Variable: CRAWLER_CANNON_TYPE (none vs scalar)
#  Scalar chosen: best raw_bpb in 1GPU gate, cheapest (3 params)
#
#  Base: BW5 (CHOKE_DIM=0, COMPILE_FULLGRAPH=1, ROPE_SCALES=9,1,1)
#
#  Pass criteria: scalar step_avg < control step_avg
#
#  Usage:
#    NPROC_PER_NODE=8 bash experiments/Bandit_Wagon_V_Cannon/gate_8gpu.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
GATE_STEPS=2000
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
LOGDIR="${REPO_ROOT}/logs"
mkdir -p "${LOGDIR}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

echo "================================================================"
echo "  BWVC CANNON — 8GPU SPEED GATE"
echo "  ${GATE_STEPS} steps | seed=${SEED} | nproc=${NPROC}"
echo "  Control vs scalar cannon — does DDP preserve speed gain?"
echo "================================================================"

BW5_BASELINE_STEP_AVG="74.68"

run_arm() {
    local arm_id="$1"
    local label="$2"
    local cannon_type="$3"

    echo ""
    echo "--- ${arm_id}: ${label} ---"
    local logfile="${LOGDIR}/bwvc_8gpu_${arm_id}_s${SEED}_$(date +%H%M%S).log"

    env \
        SEED="${SEED}" \
        ITERATIONS="${GATE_STEPS}" \
        WARMDOWN_ITERS=0 \
        MAX_WALLCLOCK_SECONDS=0 \
        COMPLEMENT_ALPHA=0 \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=2048 \
        ROPE_DIMS=16 \
        SWA_EVERY=50 \
        MTP_NUM_HEADS=0 \
        LATE_QAT_THRESHOLD=0 \
        MATRIX_LR=0.03 \
        TORCHDYNAMO_OPTIMIZE_DDP=0 \
        COMPILE_FULLGRAPH=1 \
        NGRAM_EVAL_ORDER=0 \
        MODEL_DIM=512 \
        USE_CRAWLER=1 \
        NUM_FLAT_LAYERS=4 \
        NUM_CRAWLER_LAYERS=1 \
        CRAWLER_LOOPS=3 \
        CRAWLER_MLP_MULT=6.0 \
        INST_DIM=32 \
        CRAWLER_QUANT_INT8=1 \
        DELTA_NET_HEADS=0 \
        SKIP_EMA=1 \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0 \
        NITRUST_ENABLE=0 \
        NITRUST_STRICT=0 \
        MLP_LEAKY_SLOPE=0.5 \
        CRAWLER_MLP_LEAKY_SLOPE=0.5 \
        CRAWLER_MLP_CHOKE_DIM=0 \
        CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
        CRAWLER_LOOP_SMEAR=0 \
        CRAWLER_TAP_DIM=0 \
        CRAWLER_TAP_LOOP_SPECIFIC=1 \
        CRAWLER_TAP_LAYERS=all \
        CRAWLER_CANNON_TYPE="${cannon_type}" \
        NPROC_PER_NODE="${NPROC}" \
        torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${logfile}"

    local step_avg
    step_avg=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 || echo "?")
    echo "  -> step_avg: ${step_avg}ms"
    echo "${arm_id}|${label}|${cannon_type}|${step_avg}" >> "${RESULTS_FILE}"
}

RESULTS_FILE=$(mktemp)

# Link train_gpt.py from BW5
if [[ ! -f "${SCRIPT_DIR}/train_gpt.py" ]]; then
    ln -s "${REPO_ROOT}/crawler/2026-03-29_BW5/train_gpt.py" "${SCRIPT_DIR}/train_gpt.py"
fi

run_arm BWVC-00 "control (no cannon)" none
run_arm BWVC-01 "scalar cannon (3 params)" scalar

echo ""
echo "================================================================"
echo "  BWVC 8GPU SPEED GATE SUMMARY"
echo "  BW5 full run baseline: ${BW5_BASELINE_STEP_AVG}ms/step"
echo "================================================================"
printf "%-10s %-28s %-10s %-12s\n" "ARM" "LABEL" "TYPE" "STEP_AVG"
printf "%-10s %-28s %-10s %-12s\n" "---" "-----" "----" "--------"
while IFS='|' read -r arm label cannon step_avg; do
    printf "%-10s %-28s %-10s %-12s\n" "${arm}" "${label}" "${cannon}" "${step_avg}ms"
done < "${RESULTS_FILE}"
rm -f "${RESULTS_FILE}"
echo ""
echo "  Pass: scalar step_avg < control step_avg"
echo "  If speed holds → proceed to Bandit_Wagon_V_PyramidCannon"
echo "================================================================"
