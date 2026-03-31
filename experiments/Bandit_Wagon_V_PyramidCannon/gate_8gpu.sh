#!/bin/bash
set -euo pipefail
# ================================================================
#  Bandit_Wagon_V_PyramidCannon — 8×H100 GATE
#
#  Combined hypothesis: pyramid gives cannon a calibration target,
#  cannon gives pyramid a faster compiled path.
#
#  Variable: pyramid + scalar cannon vs flat + no cannon
#  Base: BW5 (COMPILE_FULLGRAPH=1, ROPE_SCALES=9,1,1)
#  2000 steps — proper gate before full run
#
#  NOTE: Run ONLY after both individual gates pass:
#    - Bandit_Wagon_V_Cannon/gate_8gpu.sh (speed confirmed)
#    - Bandit_Wagon_V_Pyramid/gate_8gpu.sh (quality signal confirmed)
#
#  Usage:
#    NPROC_PER_NODE=8 bash experiments/Bandit_Wagon_V_PyramidCannon/gate_8gpu.sh
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
echo "  BWV PYRAMID+CANNON — 8GPU GATE"
echo "  ${GATE_STEPS} steps | seed=${SEED} | nproc=${NPROC}"
echo "  Control: flat+none | Test: pyramid+scalar"
echo "  BW5 baseline: 74.68ms/step | 1.18672 int6_sw_bpb"
echo "================================================================"

BW5_STEP_AVG="74.68"
BW5_BPB="1.18672"

run_arm() {
    local arm_id="$1"
    local label="$2"
    local choke_dim="$3"
    local choke_shape="$4"
    local cannon_type="$5"

    echo ""
    echo "--- ${arm_id}: ${label} ---"
    local logfile="${LOGDIR}/bwvpc_8gpu_${arm_id}_s${SEED}_$(date +%H%M%S).log"

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
        CRAWLER_MLP_CHOKE_DIM="${choke_dim}" \
        CRAWLER_MLP_CHOKE_SHAPE="${choke_shape}" \
        CRAWLER_MLP_CHOKE_GROUPS=8 \
        CRAWLER_CANNON_TYPE="${cannon_type}" \
        CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
        CRAWLER_LOOP_SMEAR=0 \
        CRAWLER_TAP_DIM=0 \
        CRAWLER_TAP_LOOP_SPECIFIC=1 \
        CRAWLER_TAP_LAYERS=all \
        NPROC_PER_NODE="${NPROC}" \
        torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${logfile}"

    local raw_bpb step_avg
    raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" | tail -1 || echo "?")
    step_avg=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 || echo "?")
    echo "  -> step_avg:${step_avg}ms  raw_bpb:${raw_bpb}"
    echo "${arm_id}|${label}|${choke_dim}|${cannon_type}|${step_avg}|${raw_bpb}"
}

if [[ ! -f "${SCRIPT_DIR}/train_gpt.py" ]]; then
    ln -s "${REPO_ROOT}/experiments/Bandit_Wagon_V/train_gpt.py" "${SCRIPT_DIR}/train_gpt.py"
fi

CTRL=$(run_arm BWVPC-00 "control (flat + no cannon)" 0   flat    none)
TEST=$(run_arm BWVPC-01 "pyramid + scalar cannon"    512 pyramid scalar)

echo ""
echo "================================================================"
echo "  BWV PYRAMID+CANNON 8GPU GATE SUMMARY"
echo "  seed=${SEED}  steps=${GATE_STEPS}  nproc=${NPROC}"
echo "  BW5 reference: ${BW5_STEP_AVG}ms/step | ${BW5_BPB} int6_sw_bpb (seed=444)"
echo "================================================================"
printf "%-12s %-30s %-8s %-10s %-10s %-12s\n" "ARM" "LABEL" "CHOKE" "CANNON" "STEP_AVG" "RAW_BPB"
printf "%-12s %-30s %-8s %-10s %-10s %-12s\n" "---" "-----" "-----" "------" "--------" "-------"
for r in "${CTRL}" "${TEST}"; do
    IFS='|' read -r arm label choke cannon step_avg raw <<< "${r}"
    printf "%-12s %-30s %-8s %-10s %-10s %-12s\n" "${arm}" "${label}" "${choke}" "${cannon}" "${step_avg}ms" "${raw}"
done
echo ""
echo "  Pass: BWVPC-01 raw_bpb < BWVPC-00 AND step_avg <= ${BW5_STEP_AVG}ms"
echo "  If passes → full 600s run on 8×H100"
echo "================================================================"
