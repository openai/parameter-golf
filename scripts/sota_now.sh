#!/usr/bin/env bash
# sota_now.sh — original submission approach. system python3 + hopper PYTHONPATH.
# Source: vault/train_gpt_rascal_sota_REAL.py (0ec1f462, 118521 bytes, matches seed444 log)
# Stack: cu124 required. FAIL hard on wrong env.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LOCKED_SRC="${REPO_ROOT}/vault/train_gpt_rascal_sota_REAL.py"
EXPECTED_HASH="0ec1f462ab39fd601b18f2b086f6283a0c8db3d2a9780a92dfb206ec46e067cb"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
LOG_DIR="${REPO_ROOT}/logs/sota_runs"

die() { echo "FATAL: $*" >&2; exit 1; }

# ── 1. Source hash ────────────────────────────────────────────
echo "[1/3] source hash..."
[[ -f "${LOCKED_SRC}" ]] || die "vault source not found: ${LOCKED_SRC}"
actual=$(sha256sum "${LOCKED_SRC}" | awk '{print $1}')
[[ "${actual}" == "${EXPECTED_HASH}" ]] || die "hash mismatch. got ${actual}"
echo "      OK ${actual:0:16}..."

# ── 2. CUDA must be 12.4 ─────────────────────────────────────
echo "[2/3] CUDA version (must be 12.4.x)..."
cuda_ver=$(python3 -c "import torch; print(torch.version.cuda or 'NONE')" 2>/dev/null) \
    || die "python3/torch failed — fix environment"
torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
[[ "${cuda_ver}" == "12.4"* ]] || \
    die "wrong CUDA: '${cuda_ver}' (torch ${torch_ver}). Need cu124."
echo "      torch=${torch_ver}  cuda=${cuda_ver}  OK"

# ── 3. Run — same env as original submission ──────────────────
echo "[3/3] launching (SKIP_GPTQ=1 seed=${SEED})..."
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/sota_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

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
torchrun --standalone --nproc_per_node="${NPROC}" "${LOCKED_SRC}" \
2>&1 | tee "${LOG}"

echo ""
echo "LOG: ${LOG}"
grep -E "step:500/|step:1000/|step:6[0-9]{3}/|stopping_early|final_sliding_window_exact|gptq:|Code size:" \
    "${LOG}" | tail -20 || true

# Stack parity check — must be ~91ms, abort flag if >=93ms
step500=$(grep "step:500/" "${LOG}" | grep -oP 'step_avg:\K[0-9.]+' || true)
if [[ -n "${step500}" ]]; then
    echo ""
    echo "step_avg @ 500: ${step500}ms  (record: ~90.70ms)"
    awk "BEGIN {exit !(${step500} >= 93.0)}" && true || {
        echo "STACK PARITY FAILURE — ${step500}ms >= 93ms. Wrong env. Score invalid."
        exit 2
    }
fi
