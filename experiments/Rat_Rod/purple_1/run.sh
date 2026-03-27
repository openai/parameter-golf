#!/bin/bash
set -euo pipefail
# RAT ROD PURPLE 1 — Full Stack
#
# vs Green v1 (sliding=1.1129, ngram9=0.4489):
#   MATRIX_LR=0.03          PR #859  — stronger neural base
#   WARMDOWN_ITERS=2000              — confirmed best from A/B sweep
#   NGRAM_CHUNK_TOKENS=65536 PR #850  — 15x more frequent cache refresh, kills cold-start
#   NGRAM_DIRICHLET=1       PR #900  — Bayesian posterior: p=(count+c*neural)/(ctx+c)
#   PHRASE_CACHE=1          PR #880  — variable-length suffix match (48/36/28/20/16 tok)
#   REGIME_TRACKER=1        PR #880  — adapts phrase trust for repetitive vs novel text
#
# NOT included (legally gray):
#   ARTIFACT_NGRAM=0        PR #931  — training corpus oracle; organizers haven't ruled on it
#
# Legal basis: all cache updates are score-first causal on val data only.
# Rule: score chunk → update cache. No val token ever updates cache before it is scored.

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
echo "  RAT ROD PURPLE 1 — Full Stack"
echo "  Seed: ${SEED}"
echo "  matrix_lr=0.03 | warmdown=2000"
echo "  chunk=65K | ngram_dirichlet | phrase_cache | regime_tracker"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=2000 \
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
NGRAM_CHUNK_TOKENS=65536 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=1 \
NGRAM_ORDER_MULTS="0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0" \
NGRAM_DIRICHLET=1 \
NGRAM_DIRICHLET_CONC=5.0 \
PHRASE_CACHE=1 \
PHRASE_BUCKETS=4194304 \
PHRASE_PROBE_LENGTHS="48,36,28,20,16" \
PHRASE_CONCENTRATION=2.0 \
PHRASE_MIN_COUNT=1 \
REGIME_TRACKER=1 \
ARTIFACT_NGRAM=0 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/ratrod_purple1_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
