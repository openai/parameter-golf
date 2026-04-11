#!/bin/bash
set -euo pipefail
# A-WING RED_G: Mixer-first, startup-bounded variant.
# Keeps learned mixer head, but bounds prefill and uses distributed sync
# so setup doesn't dominate runtime.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
: "${MAX_WALLCLOCK_SECONDS:=570}"

# 10-minute eval budgeting (training and eval are separate challenge caps).
: "${EVAL_BUDGET_SECONDS:=600}"
: "${EVAL_FIXED_OVERHEAD_SECONDS:=150}"
: "${EVAL_SAFETY_MARGIN_SECONDS:=45}"
DEFAULT_NGRAM_MAX_SECONDS=$((EVAL_BUDGET_SECONDS - EVAL_FIXED_OVERHEAD_SECONDS - EVAL_SAFETY_MARGIN_SECONDS))
if (( DEFAULT_NGRAM_MAX_SECONDS < 60 )); then
    DEFAULT_NGRAM_MAX_SECONDS=60
fi
: "${NGRAM_EVAL_MAX_SECONDS:=${DEFAULT_NGRAM_MAX_SECONDS}}"
: "${NGRAM_EVAL_BUCKETS:=16777216}"
: "${NGRAM_CHUNK_TOKENS:=1048576}"

# Mixer prefill controls (training-oracle build time).
: "${MIXER_BUCKETS:=2097152}"
: "${MIXER_N_ORDERS:=8}"                    # orders 2..9
: "${MIXER_PREFILL_MAX_SHARDS:=80}"
: "${MIXER_PREFILL_MAX_SECONDS:=90}"
: "${MIXER_PREFILL_MIN_SHARDS:=4}"
: "${MIXER_PREFILL_TOKENS_PER_SHARD:=50000000}"
: "${MIXER_GPU_MODE:=1}"
: "${MIXER_PREFILL_POS_CHUNK:=1000000}"

: "${COMPILE_FULLGRAPH:=0}"

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
echo "  A-WING RED_G — GPU Monster Mixer"
echo "  Seed: ${SEED}"
echo "  Mixer: Linear(512→$((MIXER_N_ORDERS + 1))) orders 2..$((MIXER_N_ORDERS + 1))"
echo "  Mixer prefill: <=${MIXER_PREFILL_MAX_SECONDS}s, min_shards=${MIXER_PREFILL_MIN_SHARDS}, max_shards=${MIXER_PREFILL_MAX_SHARDS}"
echo "  Mixer buckets: ${MIXER_BUCKETS}, tokens/shard cap: ${MIXER_PREFILL_TOKENS_PER_SHARD}, gpu_mode=${MIXER_GPU_MODE}"
echo "  Eval buckets: ${NGRAM_EVAL_BUCKETS}, ngram eval cap: ${NGRAM_EVAL_MAX_SECONDS}s"
echo "  Training cap: ${MAX_WALLCLOCK_SECONDS}s"
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
MIXER_N_ORDERS="${MIXER_N_ORDERS}" \
MIXER_LOSS_WEIGHT=0.1 \
MIXER_NEURAL_FLOOR=0.05 \
MIXER_BUCKETS="${MIXER_BUCKETS}" \
MIXER_PREFILL_MAX_SHARDS="${MIXER_PREFILL_MAX_SHARDS}" \
MIXER_PREFILL_MAX_SECONDS="${MIXER_PREFILL_MAX_SECONDS}" \
MIXER_PREFILL_MIN_SHARDS="${MIXER_PREFILL_MIN_SHARDS}" \
MIXER_PREFILL_TOKENS_PER_SHARD="${MIXER_PREFILL_TOKENS_PER_SHARD}" \
MIXER_GPU_MODE="${MIXER_GPU_MODE}" \
MIXER_PREFILL_POS_CHUNK="${MIXER_PREFILL_POS_CHUNK}" \
NGRAM_EVAL_ORDER=9 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MIN=0.05 \
NGRAM_EVAL_ALPHA_MAX=0.60 \
NGRAM_EVAL_ENTROPY_CENTER=3.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MIN_COUNT=2 \
NGRAM_EVAL_BUCKETS="${NGRAM_EVAL_BUCKETS}" \
NGRAM_EVAL_MAX_SECONDS="${NGRAM_EVAL_MAX_SECONDS}" \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=1 \
NGRAM_ORDER_MULTS="" \
NGRAM_CHUNK_TOKENS="${NGRAM_CHUNK_TOKENS}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/awing_redg_gpu_mixer_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
