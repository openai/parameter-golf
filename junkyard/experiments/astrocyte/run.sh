#!/bin/bash
set -euo pipefail
# ASTROCYTE: Tiny parallel gating network (2% params, modulates Q/K scales)
# φ bonus: hidden dims follow 1/φ geometric sequence
# Base: Green v1 stack + astrocyte module

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

ASTROCYTE_ENABLED="${ASTROCYTE_ENABLED:-1}"
ASTROCYTE_HIDDEN="${ASTROCYTE_HIDDEN:-512}"
ASTROCYTE_LR="${ASTROCYTE_LR:-0.025}"

echo "============================================"
echo "  ASTROCYTE — Tiny Parallel Gating Network"
echo "  Seed: ${SEED}"
echo "  Base: Green v1 stack + astrocyte module"
echo "  Enabled: ${ASTROCYTE_ENABLED} | Hidden: ${ASTROCYTE_HIDDEN} | LR: ${ASTROCYTE_LR}"
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
ASTROCYTE_ENABLED="${ASTROCYTE_ENABLED}" \
ASTROCYTE_HIDDEN="${ASTROCYTE_HIDDEN}" \
ASTROCYTE_LR="${ASTROCYTE_LR}" \
ASTROCYTE_LOSS_WEIGHT=0.1 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/astrocyte_s${SEED}_h${ASTROCYTE_HIDDEN}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
