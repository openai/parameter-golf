#!/bin/bash
set -euo pipefail
# RAT ROD PURPLE 1: Green base + PR #931 + PR #900 + PR #859
#
# New vs Green v1:
#   - ARTIFACT_NGRAM=1        (PR #931) — seed eval tables from 2 training shards
#                               eliminates cold-start: every val token sees a warm cache
#   - NGRAM_DIRICHLET=1       (PR #900) — Bayesian posterior mixing replaces linear alpha
#                               p = (count + c * neural) / (ctx_count + c)
#                               naturally down-weights low-count matches without hand-tuned alpha
#   - MATRIX_LR=0.03          (PR #859) — higher LR trains stronger neural model
#   - NGRAM_DIRICHLET_CONC=5  — start with flat c=5 (OBCL tuning is Purple-2)
#   - ARTIFACT_NGRAM_MAX_SHARDS=2 — process ~200M training tokens for oracle build
#
# Controls (Green v1 achieved): sliding=1.1129, ngram9=0.4489

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

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
echo "  RAT ROD PURPLE 1"
echo "  Seed: ${SEED}"
echo "  PR #931: Training oracle (warm start), 2 shards"
echo "  PR #900: Dirichlet-Multinomial mixing (c=5)"
echo "  PR #859: matrix_lr=0.03"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=600 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=1 \
LATE_QAT_THRESHOLD=0 \
MATRIX_LR=0.03 \
NGRAM_EVAL_ORDER=9 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MIN=0.05 \
NGRAM_EVAL_ALPHA_MAX=0.60 \
NGRAM_EVAL_ENTROPY_CENTER=3.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MIN_COUNT=1 \
NGRAM_EVAL_BUCKETS=8388608 \
NGRAM_EVAL_MAX_SECONDS=0 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=1 \
NGRAM_ORDER_MULTS="0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0" \
ARTIFACT_NGRAM=1 \
ARTIFACT_NGRAM_MAX_SHARDS=2 \
NGRAM_DIRICHLET=1 \
NGRAM_DIRICHLET_CONC=5.0 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/ratrod_purple1_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
