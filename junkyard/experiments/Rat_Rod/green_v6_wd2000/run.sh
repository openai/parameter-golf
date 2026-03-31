#!/bin/bash
set -euo pipefail
# RAT ROD GREEN v6: Optimized SOTA
# Base: green v1 (1.1129 sliding, 0.4489 ngram9)
# Change: WARMDOWN_ITERS=2000 (confirmed -0.0087 sliding at 200s)
# Change: LATE_QAT_THRESHOLD=0 (kills noise bug, wash but no downside)
# Change: TRIGRAM=0 (v1 default, v2 trigram was wash)
# Everything else identical to v1.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# --- Pre-flight checks ---
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

echo "============================================"
echo "  RAT ROD GREEN v6 — Optimized SOTA"
echo "  Seed: ${SEED}"
echo "  v1 base + WARMDOWN_ITERS=2000"
echo "  No GPTQ | No Siphon | No Trigram"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=600 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
LATE_QAT_THRESHOLD=0 \
WARMDOWN_ITERS=2000 \
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
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${REPO_ROOT}/experiments/Rat_Rod/green/train_gpt.py" \
    2>&1 | tee "logs/ratrod_green_v6_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
