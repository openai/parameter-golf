#!/bin/bash
set -euo pipefail
# CRAWLER_LEG_3 — Full 600s run, best Leg 2 arch, Rascal-style warmdown
#
# Leg 2 findings (8×H100, 350s):
#   At scale, quant gap collapses to ~0. Arch deltas compress vs Leg 1.
#   Best SKIP_GPTQ=1 arm: CL2-04  loops=3 + mlp=6.0  → 1.19828 BPB  (−0.0074 vs baseline)
#   Best overall:         CL2-02  loops=3 + mlp=5.0 + LOOP_AWARE_GPTQ + COMPILE → 1.19593
#
# This run: CL2-04 arch at full 600s wallclock, Rascal-style (SKIP_GPTQ=1, no GPTQ overhead).
# Goal: see how much BPB improves with 250s more training (~2400 extra steps on 8×H100).
#
# Usage:
#   NPROC_PER_NODE=8 bash experiments/Crawler_Leg_3/run.sh
#   NPROC_PER_NODE=1 bash experiments/Crawler_Leg_3/run.sh   # single-GPU test

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NITRUST_ENABLE="${NITRUST_ENABLE:-0}"
NITRUST_STRICT="${NITRUST_STRICT:-0}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

RUN_DATE="$(date +%Y%m%d_%H%M%S)"
LOG="${RESULTS_DIR}/CL3-01_${RUN_DATE}.log"

echo "============================================"
echo "  CRAWLER_LEG_3 — Full 600s, loops=3 mlp=6.0"
echo "  Seed: ${SEED}  GPUs: ${NPROC_PER_NODE}  Wallclock: 600s"
echo "  SKIP_GPTQ=1 (Rascal warmdown style)"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE}"
echo "============================================"
echo ""
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
    2>&1 | tee "${LOG}"

# --- Extract metrics ---
val_bpb=$(grep -oP 'step:\d+/\d+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
steps=$(grep -oP 'stopping_early.*step:\K[0-9]+' "${LOG}" | tail -1 \
      || grep -oP 'step:(\K[0-9]+)/20000 val_loss' "${LOG}" | tail -1 \
      || echo "?")
int6_bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")

echo ""
echo "============================================"
echo "  CL3-01  loops=3  mlp=6.0  SKIP_GPTQ=1  600s"
echo "  steps:       ${steps}"
echo "  val_bpb:     ${val_bpb}"
echo "  int6_sw_bpb: ${int6_bpb}"
echo "  Log: ${LOG}"
echo "============================================"
