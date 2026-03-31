#!/bin/bash
set -euo pipefail
# ================================================================
#  BW4 — TIER 1 FULLGRAPH GATE
#
#  Hypothesis: COMPILE_FULLGRAPH=1 compiles cleanly on BW4 (no
#  DeltaNet blocker) and reduces step_avg via kernel fusion.
#
#  BW4 baseline (COMPILE_FULLGRAPH=0): 74.80ms/step
#  Expected gain: 2-5ms/step from FLOW+block fusion, fewer
#  intermediate tensor materializations.
#
#  PASS criteria:
#    - No graph breaks / compilation errors
#    - step_avg < 74ms (any improvement counts)
#    - raw_bpb within ±0.002 of BW4 baseline at 2000 steps
#
#  IDENTICAL to gate.sh except COMPILE_FULLGRAPH=1
#
#  Usage:
#    bash experiments/Bandit_Wagon_IV/gate_fullgraph.sh
#    NPROC_PER_NODE=4 bash experiments/Bandit_Wagon_IV/gate_fullgraph.sh
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
echo "  BW4 TIER 1 — COMPILE_FULLGRAPH=1 test"
echo "  2000 steps | seed=${SEED} | nproc=${NPROC}"
echo "  Baseline step_avg: 74.80ms (COMPILE_FULLGRAPH=0)"
echo "============================================"

LOG="${LOGDIR}/bw4_fullgraph_s${SEED}_$(date +%H%M%S).log"

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
    NPROC_PER_NODE="${NPROC}" \
    torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
    2>&1 | tee "${LOG}"

# ----------------------------------------------------------------
# Extract and compare
# ----------------------------------------------------------------
int6_bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
step_avg=$(grep -oP 'step:2000/[0-9]+.*?step_avg:\K[0-9.]+' "${LOG}" | tail -1 || \
           grep -oP 'step_avg:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
graph_breaks=$(grep -c 'Graph break\|graph break\|BREAK\|TorchDynamo' "${LOG}" 2>/dev/null || echo "0")
quant_gap="?"
if [[ "${raw_bpb}" != "?" && "${int6_bpb}" != "?" ]]; then
    quant_gap=$(python3 -c "print(f'{float(\"${int6_bpb}\")-float(\"${raw_bpb}\"):.4f}')" 2>/dev/null || echo "?")
fi

echo ""
echo "============================================"
echo "  BW4 FULLGRAPH GATE RESULT"
echo "  step_avg:    ${step_avg}ms  (baseline: 74.80ms)"
echo "  raw_bpb:     ${raw_bpb}"
echo "  int6_sw_bpb: ${int6_bpb}"
echo "  quant_gap:   ${quant_gap}"
echo "  graph_break hints: ${graph_breaks}"
echo "  Log: ${LOG}"
echo ""
echo "  PASS: step_avg < 74ms AND no graph breaks"
echo "  FAIL: graph break errors OR step_avg >= 74ms"
echo "============================================"
