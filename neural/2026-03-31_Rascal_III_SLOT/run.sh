#!/usr/bin/env bash
# Rascal_III_SLOT — 8×H100 600s racer. One change vs SOTA: SLOT_ENABLED=1.
# On pod: git pull && bash neural/2026-03-31_Rascal_III_SLOT/run.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${REPO_ROOT}/neural/2026-03-31_Rascal_III_SLOT/train_gpt_slot.py"
EXPECTED_HASH="fac1d67b2779ce1b8b118284728e8799b5ab55dd43c95b38db428d3380369f17"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
LOG_DIR="${REPO_ROOT}/logs/slot_runs"

die() { echo "FATAL: $*" >&2; exit 1; }

echo "[1/3] source hash..."
actual=$(sha256sum "${SRC}" | awk '{print $1}')
[[ "${actual}" == "${EXPECTED_HASH}" ]] || die "hash mismatch — got ${actual}"
echo "      OK ${actual:0:16}..."

echo "[2/3] CUDA version (must be 12.x)..."
cuda_ver=$(python3 -c "import torch; print(torch.version.cuda or 'NONE')" 2>/dev/null) \
    || die "python3/torch failed"
torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
[[ "${cuda_ver}" == "12."* ]] || \
    die "wrong CUDA: ${cuda_ver} (torch ${torch_ver}) — requires CUDA 12.x"
echo "      torch=${torch_ver}  cuda=${cuda_ver}  OK"

echo "[3/3] launching seed=${SEED} nproc=${NPROC}..."
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/slot_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS=600 \
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

echo ""
echo "SAVE CHECKPOINT: cp \$(find ${REPO_ROOT} -name final_model.pt | head -1) ${LOG_DIR}/final_model_s${SEED}.pt"
