#!/bin/bash
set -euo pipefail
# A-WING GREEN_1A: Legal entropy-adaptive alpha + PR#609 improvements
# Changes from green_1:
#   - XSA on all 11 layers (was last 4)
#   - BigramHash 2048 (was 1536)
#   - GPTQ: descending col order, damping 0.01, block_size 128
#   - lzma compression (was zstd)
#   - Selective ±1 pruning for exact size targeting
#   - Oracle alpha REMOVED — entropy-adaptive only (legal)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# --- Pre-flight checks ---
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || echo "  WARNING: zstandard not found — using lzma (stdlib)"

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    import flash_attn; v=flash_attn.__version__
    if v.startswith('3'): print(f'  FA3 v{v} OK')
    else: print(f'  WARNING: FA{v[0]} detected — want FA3')
" 2>/dev/null || echo "  WARNING: no flash_attn found"

echo "============================================"
echo "  A-WING GREEN_1A — Legal Alpha + PR609 Improvements"
echo "  Seed: ${SEED}"
echo "  XSA-all-11, BigramHash 2048, GPTQ improved, lzma"
echo "  Entropy-adaptive alpha ONLY (no oracle)"
echo "  Training cap: 570s (30s reserved for GPTQ)"
echo "============================================"

SEED="$SEED" \
F1_CORR_RANK=0 \
DISTILL_ENABLED=0 \
MLP_ACT=leaky_relu_sq \
MLP_LEAKY_SLOPE=0.5 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
TTT_EVAL_ENABLED=0 \
ROPE_DIMS=24 \
VAL_LOSS_EVERY=20000 \
TRAIN_LOG_EVERY=1000 \
SWA_EVERY=100 \
COMPLEMENT_ALPHA=0.5 \
NGRAM_EVAL_ORDER=9 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MIN=0.05 \
NGRAM_EVAL_ALPHA_MAX=0.60 \
NGRAM_EVAL_ENTROPY_CENTER=3.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MIN_COUNT=2 \
NGRAM_EVAL_BUCKETS=8388608 \
NGRAM_EVAL_MAX_SECONDS=0 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=1 \
NGRAM_ORDER_MULTS="0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0" \
MAX_WALLCLOCK_SECONDS=570 \
COMPILE_FULLGRAPH=0 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/awing_green1A_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
