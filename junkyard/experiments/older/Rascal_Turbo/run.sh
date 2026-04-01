#!/bin/bash
set -euo pipefail
# RASCAL TURBO — Rascal II + TurboMuon (AOL + Polar NS4 + row_col post-norm)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
SEED="${SEED:-444}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

command -v "${TORCHRUN_BIN}" >/dev/null 2>&1 || { echo "ERROR: TORCHRUN_BIN not found: ${TORCHRUN_BIN}"; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: tokenizer not found: ${TOKENIZER_PATH}"; exit 1; }
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: data path not found: ${DATA_PATH}"; exit 1; }

echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null || echo "  WARNING: zstandard not found"

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    try:
        import flash_attn; v=flash_attn.__version__
        print(f'  flash_attn v{v} detected')
    except Exception:
        print('  WARNING: no flash_attn found')
" 2>/dev/null || true

echo "============================================"
echo "  RASCAL TURBO"
echo "  Seed: ${SEED}"
echo "  TurboMuon: AOL + Polar NS4 + row_col"
echo "  Wallclock: ${MAX_WALLCLOCK_SECONDS:-600}s"
echo "============================================"

mkdir -p logs

SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
SKIP_GPTQ=1 \
LOADER_MODE=coprime \
COPRIME_MAX_LOADED_SHARDS=1 \
COPRIME_SHARDS_PER_BATCH=1 \
COPRIME_SHARD_HOLD_STEPS=64 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
NGRAM_EVAL_ORDER=0 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=0 \
MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-4}" \
MUON_POST_NORM="${MUON_POST_NORM:-row_col}" \
"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/rascal_turbo_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
