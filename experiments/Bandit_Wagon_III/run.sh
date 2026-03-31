#!/bin/bash
set -euo pipefail
# ================================================================
#  Bandit_Wagon_III — PRODUCTION RUN
#
#  Crawler Leg 3 arch + pyramid-512 choke + 9,1,1 battery.
#
#  Validated findings baked in:
#    BWCS: pyramid-512 choke beats all other shapes (-0.01037 vs ctrl)
#    BWCD: 9,1,1 battery beats pyramid alone by -0.01193 at 1 shard
#          quant_gap +0.0001 (near-zero) — identical trailing loops
#
#  Zero extra parameters from the battery (RoPE scale change only).
#  Pyramid stage1 adds ~1.57M params (shared universal compression).
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=8 bash experiments/Bandit_Wagon_III/run.sh
#    SEED=300 NPROC_PER_NODE=8 bash experiments/Bandit_Wagon_III/run.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-444}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

RUN_DATE="$(date +%Y%m%d_%H%M%S)"
LOG="${RESULTS_DIR}/BW3_s${SEED}_${RUN_DATE}.log"

echo "============================================"
echo "  BANDIT_WAGON_III — 600s production"
echo "  pyramid-512 + 9,1,1 battery"
echo "  Seed: ${SEED}  GPUs: ${NPROC_PER_NODE}"
echo "============================================"
echo "  Log: ${LOG}"
echo ""

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
    CRAWLER_MLP_CHOKE_SHAPE=pyramid \
    CRAWLER_MLP_CHOKE_DIM=512 \
    CRAWLER_MLP_CHOKE_GROUPS=8 \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
    CRAWLER_LOOP_SMEAR=0 \
    CRAWLER_TAP_DIM=0 \
    CRAWLER_TAP_LOOP_SPECIFIC=1 \
    CRAWLER_TAP_LAYERS=all \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_PY}" \
    2>&1 | tee "${LOG}"

val_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
steps=$(grep -oP 'stopping_early.*step:\K[0-9]+' "${LOG}" | tail -1 \
      || grep -oP 'step:(\K[0-9]+)/20000 val_loss' "${LOG}" | tail -1 \
      || echo "?")
int6_bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")

echo ""
echo "============================================"
echo "  BW3  seed=${SEED}  pyramid-512 + 9,1,1"
echo "  steps:       ${steps}"
echo "  val_bpb:     ${val_bpb}"
echo "  int6_sw_bpb: ${int6_bpb}"
echo "  Log: ${LOG}"
echo "============================================"
