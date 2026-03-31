#!/bin/bash
set -euo pipefail
# TORNADO: EMA Teacher/Student + Legal N-gram Integration
# Base: Green v1 (1.1129 BPB SOTA) stack, unchanged
# Added: EMA teacher fires every TORNADO_CADENCE steps, produces n-gram-aware
#        KL distillation signal — student learns harder on non-n-gram-predictable tokens
# Goal: Beat green 1.1129 BPB

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# Use miniconda Python/torchrun (system torchrun is CPU-only)
export PATH="/home/frosty40/miniconda3/bin:${PATH}"
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

TORNADO_CADENCE="${TORNADO_CADENCE:-4}"
TORNADO_KL_WEIGHT="${TORNADO_KL_WEIGHT:-0.1}"
TORNADO_TEMP="${TORNADO_TEMP:-2.0}"

echo "============================================"
echo "  TORNADO — EMA Teacher/Student Oscillation"
echo "  Seed: ${SEED}"
echo "  Base: Green v1 stack (unchanged)"
echo "  Tornado cadence: ${TORNADO_CADENCE} | KL weight: ${TORNADO_KL_WEIGHT} | Temp: ${TORNADO_TEMP}"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=1 \
LATE_QAT_THRESHOLD=0 \
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
TORNADO_CADENCE="${TORNADO_CADENCE}" \
TORNADO_KL_WEIGHT="${TORNADO_KL_WEIGHT}" \
TORNADO_TEMP="${TORNADO_TEMP}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/tornado_s${SEED}_c${TORNADO_CADENCE}_kl${TORNADO_KL_WEIGHT}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
