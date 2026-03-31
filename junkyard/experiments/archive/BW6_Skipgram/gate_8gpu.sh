#!/bin/bash
set -euo pipefail
# ================================================================
#  BW6_Skipgram — 8×H100 Gate
#
#  Variable: TRIGRAM (0 vs 1)
#  Base: BW5 (COMPILE_FULLGRAPH=1, ROPE_SCALES=9,1,1, CHOKE_DIM=0)
#  Zero extra parameters — trigram hashes into existing bigram table.
#
#  Pass: TRIGRAM=1 raw_bpb < control AND step_avg within ±2ms
#
#  Usage:
#    NPROC_PER_NODE=8 bash crawler/2026-03-31_BW6_Skipgram/gate_8gpu.sh
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
echo "  BW6_SKIPGRAM — 8GPU GATE"
echo "  ${GATE_STEPS} steps | seed=${SEED} | nproc=${NPROC}"
echo "  Variable: TRIGRAM (0=bigram only vs 1=bigram+trigram)"
echo "  Zero extra parameters"
echo "  BW5 baseline: 74.68ms/step"
echo "================================================================"

BW5_STEP_AVG="74.68"
BW5_BPB="1.18672385"

run_arm() {
    local arm_id="$1"
    local label="$2"
    local trigram="$3"

    echo ""
    echo "--- ${arm_id}: ${label} ---"
    local logfile="${LOGDIR}/bw6sk_8gpu_${arm_id}_s${SEED}_$(date +%H%M%S).log"

    env \
        SEED="${SEED}" \
        ITERATIONS="${GATE_STEPS}" \
        WARMDOWN_ITERS=0 \
        MAX_WALLCLOCK_SECONDS=0 \
        COMPLEMENT_ALPHA=0 \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=2048 \
        BIGRAM_DIM=128 \
        TRIGRAM="${trigram}" \
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
        CRAWLER_CANNON_TYPE=none \
        CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
        CRAWLER_LOOP_SMEAR=0 \
        CRAWLER_TAP_DIM=0 \
        CRAWLER_TAP_LOOP_SPECIFIC=1 \
        CRAWLER_TAP_LAYERS=all \
        NPROC_PER_NODE="${NPROC}" \
        torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${logfile}"

    local raw_bpb step_avg int6_sw_bpb size_bytes
    raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" | tail -1 || echo "?")
    int6_sw_bpb=$(grep -oP 'int6_sw_bpb:\K[0-9.]+' "${logfile}" | tail -1 || echo "?")
    step_avg=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 || echo "?")
    size_bytes=$(grep -oP 'Total submission size int6\+zstd: \K[0-9]+' "${logfile}" | tail -1 || echo "?")
    echo "  -> step_avg:${step_avg}ms  raw_bpb:${raw_bpb}  int6_sw_bpb:${int6_sw_bpb}  bytes:${size_bytes}"
    echo "${arm_id}|${label}|${trigram}|${step_avg}|${raw_bpb}|${int6_sw_bpb}|${size_bytes}" >> "${RESULTS_FILE}"
}

RESULTS_FILE=$(mktemp)

run_arm BW6SK-00 "control (bigram only, TRIGRAM=0)" 0
run_arm BW6SK-01 "skipgram (bigram+trigram, TRIGRAM=1)" 1

echo ""
echo "================================================================"
echo "  BW6_SKIPGRAM 8GPU GATE SUMMARY"
echo "  seed=${SEED}  steps=${GATE_STEPS}  nproc=${NPROC}"
echo "  BW5 reference: ${BW5_STEP_AVG}ms/step | ${BW5_BPB} int6_sw_bpb (seed=444, full run)"
echo "================================================================"
printf "%-12s %-38s %-8s %-10s %-10s %-14s %-12s\n" "ARM" "LABEL" "TRIGRAM" "STEP_AVG" "RAW_BPB" "INT6_SW_BPB" "BYTES"
printf "%-12s %-38s %-8s %-10s %-10s %-14s %-12s\n" "---" "-----" "-------" "--------" "-------" "-----------" "-----"
while IFS='|' read -r arm label trigram step_avg raw int6 bytes; do
    printf "%-12s %-38s %-8s %-10s %-10s %-14s %-12s\n" "${arm}" "${label}" "${trigram}" "${step_avg}ms" "${raw}" "${int6}" "${bytes}"
done < "${RESULTS_FILE}"
rm -f "${RESULTS_FILE}"
echo ""
echo "  Pass: BW6SK-01 raw_bpb < BW6SK-00 AND step_avg within ±2ms of ${BW5_STEP_AVG}ms"
echo "  If passes → full 600s run"
echo "================================================================"
