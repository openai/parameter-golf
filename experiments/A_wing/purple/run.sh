#!/bin/bash
set -euo pipefail
# A-WING PURPLE: Learned Mixer Head — Legal N-gram Ceiling Finder
# Trains a Linear(512→12) head to predict per-token expert weights
# (neural + 11 n-gram orders 2-12). Training oracle prefilled from
# training data. Eval uses backward-looking val-data cache.
# Base: Green_1 SOTA 0.3200 BPB (neural 1.1195)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# --- Pre-flight checks ---
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || { echo "  FATAL: zstandard not found. pip install zstandard"; exit 1; }

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
echo "  A-WING PURPLE — Learned Mixer Head"
echo "  Seed: ${SEED}"
echo "  Mixer: Linear(512→12), 11 n-gram orders 2-12"
echo "  12 hash primes, INT6, no cubric"
echo "  Training cap: 570s (30s reserved for GPTQ)"
echo "============================================"

SEED="$SEED" \
F1_CORR_RANK=0 \
DISTILL_ENABLED=0 \
MLP_ACT=leaky_relu_sq \
MLP_LEAKY_SLOPE=0.5 \
XSA_LAST_N=4 \
BIGRAM_VOCAB_SIZE=1536 \
TTT_EVAL_ENABLED=0 \
ROPE_DIMS=24 \
VAL_LOSS_EVERY=20000 \
TRAIN_LOG_EVERY=1000 \
SWA_EVERY=100 \
COMPLEMENT_ALPHA=0.5 \
MIXER_ENABLED=1 \
MIXER_N_ORDERS=11 \
MIXER_LOSS_WEIGHT=0.1 \
MIXER_NEURAL_FLOOR=0.05 \
MIXER_BUCKETS=8388608 \
MIXER_PREFILL_MAX_SHARDS=20 \
NGRAM_EVAL_ORDER=12 \
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
NGRAM_ORDER_MULTS="" \
MAX_WALLCLOCK_SECONDS=570 \
COMPILE_FULLGRAPH=0 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/awing_purple_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
