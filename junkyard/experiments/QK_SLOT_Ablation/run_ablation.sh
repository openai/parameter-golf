#!/bin/bash
set -euo pipefail
# QK_SLOT single-GPU ablation launcher
# Runs 4 cases: baseline / qk_gain4 / slot_only / qk_gain4_slot
# Cross-correlates QK_GAIN_INIT=4.0 (training-side) and SLOT (eval-side)
# ~1200 steps * 4 cases, SLOT_MAX_WINDOWS=512 (~1M tokens per SLOT eval)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

SEED="${SEED:-444}"
NPROC="${NPROC:-1}"
TORCHRUN="${TORCHRUN:-torchrun}"
CASES="${CASES:-all}"

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

echo "[preflight] tokenizer path: ${TOKENIZER_PATH}"
[[ -f "${TOKENIZER_PATH}" ]] || { echo "  ERROR: tokenizer not found"; exit 1; }
echo "[preflight] data path: ${DATA_PATH}"
[[ -d "${DATA_PATH}" ]] || { echo "  ERROR: data path not found"; exit 1; }

echo "============================================"
echo "  QK_SLOT ABLATION"
echo "  Seed: ${SEED}  nproc: ${NPROC}"
echo "  Cases: ${CASES}"
echo "  1200 steps, SLOT_MAX_WINDOWS=512"
echo "============================================"

python3 "${SCRIPT_DIR}/run_ablation.py" \
    --seed "${SEED}" \
    --nproc "${NPROC}" \
    --torchrun "${TORCHRUN}" \
    --cases ${CASES}

echo "============================================"
echo "  DONE — check ${SCRIPT_DIR}/logs/"
echo "============================================"
