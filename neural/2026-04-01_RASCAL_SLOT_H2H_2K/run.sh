#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${REPO_ROOT}/neural/2026-04-01_RASCAL_SLOT_H2H_2K/train_gpt_h2h.py"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
LOG_DIR="${REPO_ROOT}/neural/2026-04-01_RASCAL_SLOT_H2H_2K/logs"
cd "${REPO_ROOT}"

die() { echo "FATAL: $*" >&2; exit 1; }

echo "[1/3] stack info..."
cuda_ver=$(python3 -c "import torch; print(torch.version.cuda or 'NONE')" 2>/dev/null) || die "python3/torch failed"
torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
if python3 -c "from flash_attn_interface import flash_attn_func" >/dev/null 2>&1; then
    fa3_status="OK"
else
    fa3_status="MISSING"
fi
echo "      torch=${torch_ver} cuda=${cuda_ver} fa3=${fa3_status}"

echo "[2/3] inputs..."
[[ -f ./data/tokenizers/fineweb_1024_bpe.model ]] || die "missing tokenizer"
ls ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin >/dev/null 2>&1 || die "missing train shards"
ls ./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin >/dev/null 2>&1 || die "missing val shards"
echo "      data/tokenizer OK"

echo "[3/3] launching 2k H2H seed=${SEED} nproc=${NPROC}..."
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/slot_h2h_2k_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED}" \
ITERATIONS=2000 \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=500 \
MAX_WALLCLOCK_SECONDS=3600 \
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
SLOT_ENABLED=1 \
SLOT_STEPS="${SLOT_STEPS:-8}" \
SLOT_LR="${SLOT_LR:-0.005}" \
torchrun --standalone --nproc_per_node="${NPROC}" "${SRC}" 2>&1 | tee "${LOG}"

echo
echo "LOG: ${LOG}"
grep -E "Serialized model|Code size|Serialized model int6\+zstd|Total submission size int6\+zstd|h2h_" "${LOG}" | tail -n 40 || true
