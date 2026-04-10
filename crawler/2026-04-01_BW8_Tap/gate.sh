#!/bin/bash
set -euo pipefail
# ================================================================
#  BW8_Tap — 1-GPU gate (2000 steps)
#
#  Variable: CRAWLER_TAP_DIM=32 CRAWLER_TAP_LOOP_SPECIFIC=0
#  Parent:   BW5 (1.18672385 BPB)
#
#  Run two arms sequentially on 1 GPU:
#    BWTA-00  control (BW5 baseline)
#    BWTA-01  shared tap dim=32
#
#  Usage:
#    bash crawler/2026-04-01_BW8_Tap/gate.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC=1
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

TORCHRUN=$(command -v torchrun 2>/dev/null || echo "python3 -m torch.distributed.run")

BASE_ENV=(
    SEED="${SEED}"
    ITERATIONS=2000
    MAX_WALLCLOCK_SECONDS=3600
    WARMDOWN_ITERS=2000
    COMPLEMENT_ALPHA=0
    XSA_LAST_N=11
    BIGRAM_VOCAB_SIZE=2048
    ROPE_DIMS=16
    SWA_EVERY=50
    MTP_NUM_HEADS=0
    LATE_QAT_THRESHOLD=0
    MATRIX_LR=0.03
    TORCHDYNAMO_OPTIMIZE_DDP=0
    COMPILE_FULLGRAPH=1
    NGRAM_EVAL_ORDER=0
    MODEL_DIM=512
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=4
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=3
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8=1
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    NITRUST_ENABLE=0
    NITRUST_STRICT=0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=0
    CRAWLER_TAP_LAYERS=all
    NPROC_PER_NODE="${NPROC}"
)

run_arm() {
    local label="$1"; shift
    local extra=("$@")
    local log="${RESULTS_DIR}/gate_${label}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "--- ARM ${label} ---"
    echo "Log: ${log}"
    env "${BASE_ENV[@]}" "${extra[@]}" \
        ${TORCHRUN} --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"
    raw=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || echo "?")
    int6=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || echo "?")
    step_ms=$(grep -oP 'step_avg\s+\K[0-9.]+' "${log}" | tail -1 || echo "?")
    bytes=$(grep -oP 'Total submission size int6\+zstd: \K[0-9]+' "${log}" | tail -1 || echo "?")
    echo ""
    echo "  ${label}: raw=${raw}  int6_sw=${int6}  step=${step_ms}ms  bytes=${bytes}"
}

echo "================================================================"
echo "  BW8_Tap — 1-GPU gate  seed=${SEED}"
echo "  Variable: CRAWLER_TAP_DIM=32 CRAWLER_TAP_LOOP_SPECIFIC=0"
echo "================================================================"

run_arm "BWTA-00" CRAWLER_TAP_DIM=0
run_arm "BWTA-01" CRAWLER_TAP_DIM=32 CRAWLER_TAP_LOOP_SPECIFIC=0

echo ""
echo "Gate complete. Compare BWTA-01 vs BWTA-00."
echo "Pass: BWTA-01 int6_sw_bpb lower than BWTA-00, step_avg ~730ms (1×H100)."
echo "Fail: regression or >750ms step time."
