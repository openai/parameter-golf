#!/bin/bash
set -euo pipefail
# ================================================================
#  Bandit_Wagon_V_Pyramid — 1GPU PROXY GATE
#
#  Variable: CRAWLER_MLP_CHOKE_DIM (0=flat vs 512=pyramid)
#  Base: BW5 (COMPILE_FULLGRAPH=1, ROPE_SCALES=9,1,1, no cannon)
#
#  Pass criteria: pyramid raw_bpb < control raw_bpb at 500 steps
#  Also watch: step_avg cost of adding pyramid
#
#  Usage:
#    bash experiments/Bandit_Wagon_V_Pyramid/gate_1gpu.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-444}"
ABLATION_STEPS="${ABLATION_STEPS:-500}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
LOGDIR="${REPO_ROOT}/logs"
mkdir -p "${LOGDIR}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

echo "================================================================"
echo "  BWV PYRAMID — 1GPU PROXY"
echo "  ${ABLATION_STEPS} steps | seed=${SEED}"
echo "  Variable: CRAWLER_MLP_CHOKE_DIM (flat=0 vs pyramid=512)"
echo "================================================================"

run_arm() {
    local arm_id="$1"
    local label="$2"
    local choke_dim="$3"
    local choke_shape="${4:-flat}"

    echo ""
    echo "--- ${arm_id}: ${label} ---"
    local logfile="${LOGDIR}/bwvp_${arm_id}_s${SEED}_$(date +%H%M%S).log"

    env \
        SEED="${SEED}" \
        ITERATIONS="${ABLATION_STEPS}" \
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
        CRAWLER_MLP_CHOKE_DIM="${choke_dim}" \
        CRAWLER_MLP_CHOKE_SHAPE="${choke_shape}" \
        CRAWLER_MLP_CHOKE_GROUPS=8 \
        CRAWLER_CANNON_TYPE=none \
        CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
        CRAWLER_LOOP_SMEAR=0 \
        CRAWLER_TAP_DIM=0 \
        CRAWLER_TAP_LOOP_SPECIFIC=1 \
        CRAWLER_TAP_LAYERS=all \
        NPROC_PER_NODE=1 \
        torchrun --standalone --nproc_per_node=1 "${TRAIN_PY}" \
        2>&1 | tee "${logfile}"

    local raw_bpb step_avg
    raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" | tail -1 || echo "?")
    step_avg=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 || echo "?")
    echo "  -> step_avg:${step_avg}ms  raw_bpb:${raw_bpb}"
    echo "${arm_id}|${label}|${choke_dim}|${step_avg}|${raw_bpb}" >> "${RESULTS_FILE}"
}

RESULTS_FILE=$(mktemp)

if [[ ! -f "${SCRIPT_DIR}/train_gpt.py" ]]; then
    ln -s "${REPO_ROOT}/experiments/Bandit_Wagon_V/train_gpt.py" "${SCRIPT_DIR}/train_gpt.py"
fi

run_arm BWVP-00 "control (flat, CHOKE_DIM=0)" 0   flat
run_arm BWVP-01 "pyramid (CHOKE_DIM=512)"     512 pyramid

echo ""
echo "================================================================"
echo "  BWV PYRAMID 1GPU SUMMARY"
echo "  seed=${SEED}  steps=${ABLATION_STEPS}"
echo "================================================================"
printf "%-10s %-30s %-10s %-10s %-12s\n" "ARM" "LABEL" "CHOKE" "STEP_AVG" "RAW_BPB"
printf "%-10s %-30s %-10s %-10s %-12s\n" "---" "-----" "-----" "--------" "-------"
while IFS='|' read -r arm label choke step_avg raw; do
    printf "%-10s %-30s %-10s %-10s %-12s\n" "${arm}" "${label}" "${choke}" "${step_avg}ms" "${raw}"
done < "${RESULTS_FILE}"
rm -f "${RESULTS_FILE}"
echo ""
echo "  Pass: pyramid raw_bpb < control. Also note step_avg cost."
echo "  If step_avg cost > +5ms → pyramid too slow for 600s budget."
echo "================================================================"
