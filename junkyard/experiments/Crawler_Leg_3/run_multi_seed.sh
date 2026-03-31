#!/bin/bash
set -euo pipefail
# CRAWLER_LEG_3 — Multi-seed confirmation runs
#
# Config: loops=3, mlp=6.0, SKIP_GPTQ=1, 600s  (CL3-01 baseline = seed 1337 → 1.18720 BPB)
# This script runs seeds 42 and 300 sequentially for 3-seed submission prep.
#
# Usage:
#   NPROC_PER_NODE=8 bash experiments/Crawler_Leg_3/run_multi_seed.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NITRUST_ENABLE="${NITRUST_ENABLE:-0}"
NITRUST_STRICT="${NITRUST_STRICT:-0}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

RUN_DATE="$(date +%Y%m%d_%H%M%S)"
SUMMARY_TMP="${RESULTS_DIR}/multiseed_${RUN_DATE}.tmp"
SUMMARY="${RESULTS_DIR}/multiseed_${RUN_DATE}.txt"

echo "============================================"
echo "  CRAWLER_LEG_3 — Multi-seed (42, 300)"
echo "  Config: loops=3 mlp=6.0 SKIP_GPTQ=1 600s"
echo "  GPUs: ${NPROC_PER_NODE}"
echo "============================================"
echo ""

run_seed() {
    local seed="$1"
    local log="${RESULTS_DIR}/CL3-s${seed}_${RUN_DATE}.log"

    echo "================================================================"
    echo "  SEED ${seed}"
    echo "  Log: ${log}"
    echo "================================================================"

    env \
        SEED="${seed}" \
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
        CRAWLER_LOOPS=3 \
        CRAWLER_MLP_MULT=6.0 \
        INST_DIM=32 \
        CRAWLER_QUANT_INT8=1 \
        DELTA_NET_HEADS=0 \
        SKIP_EMA=1 \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0 \
        NITRUST_ENABLE="${NITRUST_ENABLE}" \
        NITRUST_STRICT="${NITRUST_STRICT}" \
        NITRUST_SO_PATH="${NITRUST_SO_PATH}" \
        torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
            "${REPO_ROOT}/experiments/Medusa/train_gpt.py" \
        2>&1 | tee "${log}"

    local int6_bpb steps bytes_total
    int6_bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || echo "?")
    steps=$(grep -oP 'stopping_early.*step:\K[0-9]+' "${log}" | tail -1 || echo "?")
    bytes_total=$(grep -oP 'Total submission size int6\+zstd: \K[0-9]+' "${log}" | tail -1 || echo "?")

    echo "seed_${seed}|${int6_bpb}|${steps}|${bytes_total}" >> "${SUMMARY_TMP}"
    echo "  -> seed ${seed}: int6_sw_bpb=${int6_bpb}  steps=${steps}  bytes=${bytes_total}"
    echo ""
}

run_seed 42
run_seed 300

echo "================================================================"
echo "  MULTI-SEED SUMMARY"
echo "  (seed 1337 = 1.18720375, 8087 steps, 8842981 bytes — from CL3-01)"
printf "  %-12s %14s %8s %12s\n" "SEED" "INT6_SW_BPB" "STEPS" "BYTES"
printf "  %-12s %14s %8s %12s\n" "----" "-----------" "-----" "-----"
printf "  %-12s %14s %8s %12s\n" "1337 (done)" "1.18720375" "8087" "8842981"
while IFS='|' read -r seed bpb steps bytes; do
    printf "  %-12s %14s %8s %12s\n" "${seed}" "${bpb}" "${steps}" "${bytes}"
done < "${SUMMARY_TMP}"
echo "================================================================"

mv "${SUMMARY_TMP}" "${SUMMARY}"
echo "  Full logs: ${RESULTS_DIR}/"
echo "================================================================"
