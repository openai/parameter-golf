#!/bin/bash
set -euo pipefail
# GREENROD X_1: Hybrid GDN + Standard Attention + Our Full Stack
# Architecture: First 6 layers GatedDeltaNet (fla), last 5 standard attention
# Stack: Parallel Muon + XSA-all-5 + BigramHash + Trigram + N-gram eval
# Hypothesis: DeltaNet early layers give ~0.09 BPB base model gain (PR#875)
# while our attention stack + Muon + n-gram eval add on top

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# --- Pre-flight checks ---
echo "[preflight] checking fla..."
python3 -c "from fla.layers.delta_net import DeltaNet; print('  fla DeltaNet OK')" 2>/dev/null \
    || { echo "  FATAL: fla not found — pip install flash-linear-attention fla-core"; exit 1; }

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
echo "  GREENROD X_1 — Hybrid GDN + Attention"
echo "  Seed: ${SEED}"
echo "  6x DeltaNet + 5x Standard Attention"
echo "  + Parallel Muon, XSA, Trigram, N-gram eval"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=600 \
GDN_NUM_LAYERS=6 \
GDN_LR=0.0018 \
XSA_LAST_N=5 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=1 \
LATE_QAT_THRESHOLD=0 \
COMPILE_ENABLED=1 \
COMPILE_FULLGRAPH=0 \
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
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/greenrod_x1_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
