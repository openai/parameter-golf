#!/usr/bin/env bash
set -euo pipefail
# X-WING cubric × n-gram delta sweep (eval-only).
# Requires an existing quantized model (int6 .ptz), no retraining.
#
# Usage:
#   MODEL_PATH=final_model.int6.ptz NPROC_PER_NODE=8 bash concepts/xwing/run_delta_sweep.sh
#   DELTA_GRID=interaction4 SWEEP_MAX_SECONDS=120 bash concepts/xwing/run_delta_sweep.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-final_model.int6.ptz}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SWEEP_MAX_SECONDS="${SWEEP_MAX_SECONDS:-180}"
DELTA_GRID="${DELTA_GRID:-delta12}"             # interaction4 | delta12
CUBRIC_CADENCE="${CUBRIC_CADENCE:-32}"
SWEEP_RESULTS="${SWEEP_RESULTS:-sweep_cubric_ngram_delta_results.csv}"
SWEEP_SUMMARY="${SWEEP_SUMMARY:-sweep_cubric_ngram_delta_summary.json}"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}"
    exit 1
fi

echo "============================================"
echo "  X-WING CUBRIC × NGRAM DELTA SWEEP"
echo "  Model: ${MODEL_PATH}"
echo "  Grid: ${DELTA_GRID}"
echo "  Per-ngram arm budget: ${SWEEP_MAX_SECONDS}s"
echo "  Cubric cadence (enabled arms): ${CUBRIC_CADENCE}"
echo "  GPUs: ${NPROC_PER_NODE}"
echo "============================================"

# Architecture env must match training recipe used for the model.
SEED="${SEED:-1337}" \
MLP_ACT=leaky_relu_sq \
MLP_LEAKY_SLOPE=0.5 \
XSA_LAST_N=4 \
BIGRAM_VOCAB_SIZE=1536 \
ROPE_DIMS=24 \
TTT_EVAL_ENABLED=0 \
COMPILE_ENABLED="${COMPILE_ENABLED:-0}" \
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}" \
MODEL_PATH="${MODEL_PATH}" \
SWEEP_MAX_SECONDS="${SWEEP_MAX_SECONDS}" \
DELTA_GRID="${DELTA_GRID}" \
CUBRIC_CADENCE="${CUBRIC_CADENCE}" \
SWEEP_RESULTS="${SWEEP_RESULTS}" \
SWEEP_SUMMARY="${SWEEP_SUMMARY}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/sweep_cubric_ngram_delta.py" \
    2>&1 | tee "logs/sweep_cubric_ngram_delta_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "============================================"
echo "  DELTA SWEEP DONE"
echo "  CSV: ${SWEEP_RESULTS}"
echo "  JSON: ${SWEEP_SUMMARY}"
echo "============================================"

