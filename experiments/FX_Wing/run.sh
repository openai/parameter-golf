#!/bin/bash
set -euo pipefail
# FX-WING: Instructed Recurrence + SOTA eval stack
#
# Architecture: F-Wing CrawlerGPT with inst_dim=32 instructed recurrence.
#   Content-derived per-token, per-iteration instructions from the flat encoder
#   replace fixed orthogonal loop_pos offsets, fixing the Frugendorff/CrawlerGPT
#   shared-weight gradient conflict.
#
# Training base: Rat Rod Green SOTA config
#   (Parallel Muon + XSA-all-11 + Trigram + entropy-adaptive ngram eval)
#
# Eval stack: Rat Rod Purple-1
#   matrix_lr=0.03 | warmdown=2000 | chunk=65K
#   ngram_dirichlet | phrase_cache | regime_tracker
#
# Crawler arch: 4 flat layers (U-Net enc/dec) + 1 crawler layer x 2 loops
# Legal basis: all cache updates are score-first causal on val data only.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
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
echo "  FX-WING — Instructed Recurrence + Purple eval"
echo "  Seed: ${SEED}"
echo "  inst_dim=32 | 4 flat + 1 crawler x 2 loops"
echo "  matrix_lr=0.03 | warmdown=2000 | chunk=65K"
echo "  ngram_dirichlet | phrase_cache | regime_tracker"
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
TORCHDYNAMO_OPTIMIZE_DDP=0 \
COMPILE_FULLGRAPH=0 \
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
USE_CRAWLER=1 \
NUM_FLAT_LAYERS=4 \
NUM_CRAWLER_LAYERS=1 \
CRAWLER_LOOPS=4 \
INST_DIM=32 \
CRAWLER_QUANT_INT8=1 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/fxwing_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
