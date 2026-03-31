#!/bin/bash
set -euo pipefail
# ================================================================
#  Bandit_Wagon_III — GATE VALIDATION
#
#  2000-step signal check before committing to 8×H100.
#  Pyramid-512 + 9,1,1 battery on Crawler Leg 3 arch.
#
#  Must beat Leg 3 proxy reference to proceed.
#  Reference: Leg 3 at 500 steps (1 shard) ≈ 1.447 raw_bpb
#
#  Usage:
#    bash experiments/Bandit_Wagon_III/gate.sh
#    NPROC_PER_NODE=4 bash experiments/Bandit_Wagon_III/gate.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
LOGDIR="${REPO_ROOT}/logs"
mkdir -p "${LOGDIR}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

echo "============================================"
echo "  BW3 GATE — pyramid-512 + 9,1,1 battery"
echo "  2000 steps | seed=${SEED} | nproc=${NPROC}"
echo "============================================"

LOG="${LOGDIR}/bw3_gate_s${SEED}_$(date +%H%M%S).log"

env \
    SEED="${SEED}" \
    ITERATIONS=2000 \
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
    NPROC_PER_NODE="${NPROC}" \
    torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
    2>&1 | tee "${LOG}"

int6_bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
quant_gap="?"
if [[ "${raw_bpb}" != "?" && "${int6_bpb}" != "?" ]]; then
    quant_gap=$(python3 -c "print(f'{float(\"${int6_bpb}\")-float(\"${raw_bpb}\"):.4f}')" 2>/dev/null || echo "?")
fi

echo ""
echo "============================================"
echo "  BW3 GATE RESULT"
echo "  raw_bpb:     ${raw_bpb}"
echo "  int6_sw_bpb: ${int6_bpb}"
echo "  quant_gap:   ${quant_gap}"
echo "  Log: ${LOG}"
echo "  PROCEED to 8xH100 if int6_sw_bpb is clearly"
echo "  below Leg 3 gate reference."
echo "============================================"
