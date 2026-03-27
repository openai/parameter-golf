#!/bin/bash
set -euo pipefail
# CAMBRIAN: GatedDeltaNet × Bio Seam Architecture
# Base: Green v1 (1.1129 BPB SOTA) stack
# Added: GatedDeltaNet recurrent blocks (bottom N layers) replacing CausalSelfAttention
# Goal: Beat PR#875 (1.0226 BPB) using chunked recurrence + our Muon + XSA + n-gram stack
# Cambrian-0: pure GatedDeltaNet baseline, no bio seam controllers yet

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# Use miniconda Python/torchrun (system torchrun is CPU-only)
export PATH="/home/frosty40/miniconda3/bin:${PATH}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CAMBRIAN_DELTA_LAYERS="${CAMBRIAN_DELTA_LAYERS:-6}"

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
echo "  CAMBRIAN — GatedDeltaNet × Bio Seam Architecture"
echo "  Seed: ${SEED}"
echo "  Delta layers: ${CAMBRIAN_DELTA_LAYERS} (bottom N use GDN, top layers use XSA attention)"
echo "  Stack: Muon + XSA + Trigram + N-gram eval"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}" \
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
CAMBRIAN_DELTA_LAYERS="${CAMBRIAN_DELTA_LAYERS}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/cambrian_s${SEED}_dl${CAMBRIAN_DELTA_LAYERS}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
