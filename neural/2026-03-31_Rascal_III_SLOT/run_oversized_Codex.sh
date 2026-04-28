#!/usr/bin/env bash
# Rascal_III_SLOT — 8×H100 600s racer. One change vs SOTA: SLOT_ENABLED=1.
# On pod: git pull && bash neural/2026-03-31_Rascal_III_SLOT/run.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${REPO_ROOT}/neural/2026-03-31_Rascal_III_SLOT/train_gpt_slot_oversized_Codex.py"
EXPECTED_HASH="1bd936007f7cc38f5bfe967025b396a439b0b908c22deced53f90bb9a6f08e8b"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
EXPECTED_MODEL_PARAMS="${EXPECTED_MODEL_PARAMS:-26993756}"
EXPECTED_TIE_EMBEDDINGS="${EXPECTED_TIE_EMBEDDINGS:-True}"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
LOG_DIR="${REPO_ROOT}/logs/slot_runs"
REQUIRED_TORCH_VERSION="${REQUIRED_TORCH_VERSION:-2.4.1+cu124}"
REQUIRED_CUDA_PREFIX="${REQUIRED_CUDA_PREFIX:-12.4}"
REQUIRE_FA3="${REQUIRE_FA3:-1}"
SKIP_GPTQ="${SKIP_GPTQ:-1}"
GPTQ_RESERVE_MS="${GPTQ_RESERVE_MS:-30000}"
FA3_DEFAULT_PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
FA3_PYTHONPATH="${FA3_PYTHONPATH:-}"

die() { echo "FATAL: $*" >&2; exit 1; }

echo "[1/3] source hash..."
actual=$(sha256sum "${SRC}" | awk '{print $1}')
[[ "${actual}" == "${EXPECTED_HASH}" ]] || die "hash mismatch — got ${actual}"
echo "      OK ${actual:0:16}..."

echo "[2/3] CUDA must be cu124 (SOTA stack)..."
cuda_ver=$(python3 -c "import torch; print(torch.version.cuda or 'NONE')" 2>/dev/null) \
    || die "python3/torch failed"
torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
[[ "${cuda_ver}" == "${REQUIRED_CUDA_PREFIX}"* ]] || \
    die "wrong CUDA: ${cuda_ver} (torch ${torch_ver}) — SOTA requires ${REQUIRED_CUDA_PREFIX}x. Run: bash scripts/pod_setup.sh"
[[ "${torch_ver}" == "${REQUIRED_TORCH_VERSION}" ]] || \
    die "wrong torch: ${torch_ver} — SOTA requires ${REQUIRED_TORCH_VERSION}. Run: bash scripts/pod_setup.sh"
if [[ "${REQUIRE_FA3}" == "1" ]]; then
    if [[ -n "${FA3_PYTHONPATH}" ]]; then
        PYTHONPATH="${FA3_PYTHONPATH}" python3 -c "from flash_attn_interface import flash_attn_func; print('fa3_ok')" >/dev/null 2>&1 \
            || die "FA3 import failed under FA3_PYTHONPATH=${FA3_PYTHONPATH}"
    elif PYTHONPATH="${FA3_DEFAULT_PYTHONPATH}" python3 -c "from flash_attn_interface import flash_attn_func; print('fa3_ok')" >/dev/null 2>&1; then
        FA3_PYTHONPATH="${FA3_DEFAULT_PYTHONPATH}"
    elif python3 -c "from flash_attn_interface import flash_attn_func; print('fa3_ok')" >/dev/null 2>&1; then
        FA3_PYTHONPATH="${PYTHONPATH:-}"
    else
        die "flash_attn_interface missing or ABI-mismatched (e.g. undefined symbol). Rebuild/install FA3 for torch=${torch_ver} cuda=${cuda_ver}."
    fi
fi
echo "      torch=${torch_ver}  cuda=${cuda_ver}  OK"

echo "[3/3] launching seed=${SEED} nproc=${NPROC}..."
[[ -d "${DATA_PATH}" ]] || die "DATA_PATH not found: ${DATA_PATH}"
[[ -f "${TOKENIZER_PATH}" ]] || die "TOKENIZER_PATH not found: ${TOKENIZER_PATH}"
cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/slot_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
export PYTHONPATH="${FA3_PYTHONPATH:-${PYTHONPATH:-}}"

SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS=600 \
SKIP_GPTQ="${SKIP_GPTQ}" \
GPTQ_RESERVE_MS="${GPTQ_RESERVE_MS}" \
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
DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3.0 \
TIE_EMBEDDINGS=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
torchrun --standalone --nproc_per_node="${NPROC}" "${SRC}" \
2>&1 | tee "${LOG}"

echo ""
echo "LOG: ${LOG}"
grep -E "step:500/|stopping_early|final_sliding_window|final_int6_roundtrip_exact|Total submission size" \
    "${LOG}" | tail -20 || true

step500=$(grep "step:500/" "${LOG}" | grep -oP 'step_avg:\K[0-9.]+' || true)
if [[ -n "${step500}" ]]; then
    echo "step_avg @ 500: ${step500}ms  (record: ~90.70ms)"
    if awk "BEGIN {exit (${step500} < 93.0 ? 1 : 0)}"; then
        echo "STACK PARITY FAILURE — ${step500}ms >= 93ms. Wrong env."
        exit 2
    fi
fi

model_params=$(grep -m1 "model_params:" "${LOG}" | awk -F: '{print $2}' | tr -d '[:space:]' || true)
tie_embeddings=$(grep -m1 "tie_embeddings:" "${LOG}" | sed -n 's/.*tie_embeddings:\([^ ]*\).*/\1/p' || true)
if [[ -z "${model_params}" || -z "${tie_embeddings}" ]]; then
    die "missing model config lines in log (model_params/tie_embeddings)"
fi
if [[ "${model_params}" != "${EXPECTED_MODEL_PARAMS}" ]]; then
    die "model_params drift: got ${model_params}, expected ${EXPECTED_MODEL_PARAMS} (likely leaked env var changed model shape)"
fi
if [[ "${tie_embeddings}" != "${EXPECTED_TIE_EMBEDDINGS}" ]]; then
    die "tie_embeddings drift: got ${tie_embeddings}, expected ${EXPECTED_TIE_EMBEDDINGS} (this can add hundreds of KB)"
fi
echo "config parity: model_params=${model_params} tie_embeddings=${tie_embeddings} OK"
grep -m1 "train_loader:dataset" "${LOG}" || true
grep -m1 "val_loader:shards pattern" "${LOG}" || true

echo ""
echo "SAVE CHECKPOINT: cp \$(find ${REPO_ROOT} -name final_model.pt | head -1) ${LOG_DIR}/final_model_s${SEED}.pt"
