#!/bin/bash
set -euo pipefail
# CRAWLER_LEG_1: crawler-only research lane (DeltaNet quarantined)
#
# Policy:
# - DELTA_NET_HEADS=0 (always off in this lane)
# - SKIP_GPTQ=1 for fast, stable crawler signal collection
# - LOOP_AWARE_GPTQ=0 (delta/GPTQ interaction out of scope here)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NITRUST_ENABLE="${NITRUST_ENABLE:-1}"
NITRUST_STRICT="${NITRUST_STRICT:-1}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"

NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-4}"
NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-1}"
CRAWLER_LOOPS="${CRAWLER_LOOPS:-4}"
INST_DIM="${INST_DIM:-32}"
CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8:-1}"
CRAWLER_MLP_MULT="${CRAWLER_MLP_MULT:-4.0}"

echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || echo "  WARNING: zstandard not found"

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    import flash_attn; v=flash_attn.__version__
    if v.startswith('3'): print(f'  FA3 v{v} OK')
    else: print(f'  WARNING: FA{v[0]} detected — want FA3')
" 2>/dev/null || echo "  WARNING: no flash_attn found"

if [ "${NITRUST_ENABLE}" = "1" ]; then
    if [ -f "${NITRUST_SO_PATH}" ]; then
        echo "[preflight] nitrust_py found: ${NITRUST_SO_PATH}"
    else
        if [ "${NITRUST_STRICT}" = "1" ]; then
            echo "[preflight] FATAL: NITRUST_ENABLE=1 but missing ${NITRUST_SO_PATH}"
            exit 1
        fi
        echo "[preflight] WARNING: missing ${NITRUST_SO_PATH}; run will fall back to Python path"
    fi
fi

echo "============================================"
echo "  CRAWLER_LEG_1 — Delta OFF"
echo "  Seed: ${SEED}"
echo "  flat=${NUM_FLAT_LAYERS} crawler_layers=${NUM_CRAWLER_LAYERS} loops=${CRAWLER_LOOPS}"
echo "  inst_dim=${INST_DIM} crawler_quant_int8=${CRAWLER_QUANT_INT8} crawler_mlp_mult=${CRAWLER_MLP_MULT}"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE} NITRUST_STRICT=${NITRUST_STRICT}"
echo "============================================"

SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
WARMDOWN_ITERS="${WARMDOWN_ITERS:-2000}" \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N="${XSA_LAST_N:-11}" \
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}" \
ROPE_DIMS="${ROPE_DIMS:-16}" \
SWA_EVERY="${SWA_EVERY:-50}" \
MTP_NUM_HEADS=0 \
LATE_QAT_THRESHOLD=0 \
MATRIX_LR="${MATRIX_LR:-0.03}" \
TORCHDYNAMO_OPTIMIZE_DDP="${TORCHDYNAMO_OPTIMIZE_DDP:-0}" \
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}" \
NGRAM_EVAL_ORDER=0 \
USE_CRAWLER=1 \
NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS}" \
NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS}" \
CRAWLER_LOOPS="${CRAWLER_LOOPS}" \
CRAWLER_MLP_MULT="${CRAWLER_MLP_MULT}" \
INST_DIM="${INST_DIM}" \
CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8}" \
DELTA_NET_HEADS=0 \
SKIP_EMA=1 \
SKIP_GPTQ=1 \
LOOP_AWARE_GPTQ=0 \
NITRUST_ENABLE="${NITRUST_ENABLE}" \
NITRUST_STRICT="${NITRUST_STRICT}" \
NITRUST_SO_PATH="${NITRUST_SO_PATH}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${REPO_ROOT}/experiments/Medusa/train_gpt.py" \
    2>&1 | tee "logs/crawler_leg1_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
