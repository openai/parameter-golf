#!/bin/bash
set -euo pipefail
# RAT ROD SIPHON: Ensemble-objective training
# Base: green v1 (PR#609 Parallel Muon + Parameter Banking + XSA-all)
# Added: Siphon — train on -log(α·p_ngram + (1-α)·p_model) instead of -log(p_model)
# GPU-side bigram count tables, zero new params, ~0.3ms overhead
# Goal: Model specializes away from n-gram predictions → better ensemble

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
COMPILE_ENABLED="${COMPILE_ENABLED:-1}"
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"
TORCHDYNAMO_SUPPRESS_ERRORS="${TORCHDYNAMO_SUPPRESS_ERRORS:-1}"

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
echo "  RAT ROD SIPHON — Ensemble Loss Training"
echo "  Seed: ${SEED}"
echo "  Siphon: α=0.50, 2M buckets, max_p=0.8, WD=2000"
echo "  Base: Parallel Muon, XSA-all-11, No GPTQ"
echo "  B-WING n-gram eval | QAT killed"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=600 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
LATE_QAT_THRESHOLD=0 \
WARMDOWN_ITERS=2000 \
SIPHON_ENABLED=1 \
SIPHON_ALPHA=0.50 \
SIPHON_BUCKETS=2097152 \
SIPHON_MAX_P=0.8 \
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
COMPILE_ENABLED="${COMPILE_ENABLED}" \
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH}" \
TORCHDYNAMO_SUPPRESS_ERRORS="${TORCHDYNAMO_SUPPRESS_ERRORS}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/ratrod_siphon_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
