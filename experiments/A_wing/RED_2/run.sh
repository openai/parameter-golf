#!/bin/bash
set -euo pipefail
# A-WING RED_2: legal n-gram frontier stack from GREEN backbone.
# Core strategy: entropy-gated multi-order backoff + logit-domain mixing +
# fixed-share expert tracking (non-stationary order adaptation).

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

# RED_2 evaluation mixer defaults (legal/no-oracle).
: "${NGRAM_USE_LEARNED_ALPHA:=0}"
: "${NGRAM_EVAL_ALPHA_CLIP:=0.95}"
: "${NGRAM_ENTROPY_SHIFT_PER_ORDER:=0.25}"
: "${NGRAM_ORDER_MULTS:=0.30,0.30,0.97,2.00,2.00,2.00,2.00,2.00}"
: "${NGRAM_LOGIT_MIX:=1}"
: "${NGRAM_LOGIT_MIX_EPS:=0.000001}"
: "${NGRAM_FIXED_SHARE_GAMMA:=0.015}"
: "${NGRAM_FIXED_SHARE_ETA:=0.080}"
: "${NGRAM_FIXED_SHARE_MIN_CHUNK_TOKENS:=4096}"

# Complementary training defaults.
: "${COMPLEMENT_ALPHA:=0.55}"
: "${COMPLEMENT_NOISE_FLOOR:=3}"
: "${COMPLEMENT_NOISE_WEIGHT:=0.85}"

# Learned mixer is available but disabled by default for stability.
: "${MIXER_ENABLED:=0}"
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
echo "  A-WING RED_2 — Legal Hybrid Mixer"
echo "  Seed: ${SEED}"
echo "  Blend: entropy-gated + logit-mix=${NGRAM_LOGIT_MIX}"
echo "  Fixed-Share: gamma=${NGRAM_FIXED_SHARE_GAMMA}, eta=${NGRAM_FIXED_SHARE_ETA}"
echo "  Eval buckets: ${NGRAM_EVAL_BUCKETS}, ngram cap: ${NGRAM_EVAL_MAX_SECONDS}s"
echo "  Learned mixer enabled: ${MIXER_ENABLED} (default off)"
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
COMPLEMENT_ALPHA="${COMPLEMENT_ALPHA}" \
COMPLEMENT_NOISE_FLOOR="${COMPLEMENT_NOISE_FLOOR}" \
COMPLEMENT_NOISE_WEIGHT="${COMPLEMENT_NOISE_WEIGHT}" \
MIXER_ENABLED="${MIXER_ENABLED}" \
NGRAM_EVAL_ORDER=9 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MIN=0.05 \
NGRAM_EVAL_ALPHA_MAX=0.60 \
NGRAM_EVAL_ALPHA_CLIP="${NGRAM_EVAL_ALPHA_CLIP}" \
NGRAM_EVAL_ENTROPY_CENTER=3.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MIN_COUNT=2 \
NGRAM_EVAL_BUCKETS="${NGRAM_EVAL_BUCKETS}" \
NGRAM_EVAL_MAX_SECONDS="${NGRAM_EVAL_MAX_SECONDS}" \
NGRAM_USE_LEARNED_ALPHA="${NGRAM_USE_LEARNED_ALPHA}" \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=1 \
NGRAM_ENTROPY_SHIFT_PER_ORDER="${NGRAM_ENTROPY_SHIFT_PER_ORDER}" \
NGRAM_ORDER_MULTS="${NGRAM_ORDER_MULTS}" \
NGRAM_LOGIT_MIX="${NGRAM_LOGIT_MIX}" \
NGRAM_LOGIT_MIX_EPS="${NGRAM_LOGIT_MIX_EPS}" \
NGRAM_FIXED_SHARE_GAMMA="${NGRAM_FIXED_SHARE_GAMMA}" \
NGRAM_FIXED_SHARE_ETA="${NGRAM_FIXED_SHARE_ETA}" \
NGRAM_FIXED_SHARE_MIN_CHUNK_TOKENS="${NGRAM_FIXED_SHARE_MIN_CHUNK_TOKENS}" \
NGRAM_CHUNK_TOKENS="${NGRAM_CHUNK_TOKENS}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/awing_red2_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
