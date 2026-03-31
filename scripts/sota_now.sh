#!/usr/bin/env bash
# sota_now.sh — one shot, right stack, FAIL on anything wrong.
# torch==2.5.1+cu124  |  SKIP_GPTQ=1  |  locked source hash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LOCKED_SRC="records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py"
EXPECTED_HASH="7b5bffe2601ff2fa54829a0b5b5dff7a5ad51894f2ea5a923a952c1477c7bfc6"
VENV="${REPO_ROOT}/.venv-sota"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
LOG_DIR="${REPO_ROOT}/logs/sota_runs"

die() { echo "FATAL: $*" >&2; exit 1; }

# ── 1. Source hash ────────────────────────────────────────────
echo "[1/5] source hash..."
[[ -f "${LOCKED_SRC}" ]] || die "source not found: ${LOCKED_SRC}"
actual=$(sha256sum "${LOCKED_SRC}" | awk '{print $1}')
[[ "${actual}" == "${EXPECTED_HASH}" ]] || die "hash mismatch. got: ${actual}"
echo "      OK ${actual:0:16}..."

# ── 2. Build clean venv with torch+cu124 ─────────────────────
echo "[2/5] building .venv-sota with torch==2.5.1+cu124..."
rm -rf "${VENV}"
python3 -m venv "${VENV}"
"${VENV}/bin/pip" install -q --upgrade pip
"${VENV}/bin/pip" install -q \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.5.1"
"${VENV}/bin/pip" install -q numpy zstandard sentencepiece
echo "      done"

# ── 3. Hard verify CUDA==12.4 ─────────────────────────────────
echo "[3/5] verifying CUDA==12.4..."
cuda_ver=$("${VENV}/bin/python" -c "import torch; print(torch.version.cuda or 'NONE')")
torch_ver=$("${VENV}/bin/python" -c "import torch; print(torch.__version__)")
[[ "${cuda_ver}" == "12.4"* ]] || \
    die "wrong CUDA: ${cuda_ver} (torch ${torch_ver}). pip install failed or index wrong."
echo "      torch=${torch_ver}  cuda=${cuda_ver}  OK"

# ── 4. GPU count ──────────────────────────────────────────────
echo "[4/5] GPU count..."
gpu_count=$("${VENV}/bin/python" -c "import torch; print(torch.cuda.device_count())")
[[ "${gpu_count}" -gt 0 ]] || die "no GPUs visible"
echo "      ${gpu_count} GPU(s)"

# ── 5. Run ────────────────────────────────────────────────────
echo "[5/5] running locked Rascal baseline (SKIP_GPTQ=1 seed=${SEED})..."
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/sota_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

SKIP_GPTQ=1 \
SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS=600 \
LOADER_MODE=coprime \
COPRIME_MAX_LOADED_SHARDS=1 \
COPRIME_SHARDS_PER_BATCH=1 \
COPRIME_SHARD_HOLD_STEPS=64 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=128 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
NGRAM_EVAL_ORDER=0 \
MTP_NUM_HEADS=0 \
"${VENV}/bin/python" -m torch.distributed.run \
    --standalone --nproc_per_node="${NPROC}" \
    "${LOCKED_SRC}" \
2>&1 | tee "${LOG}"

echo ""
echo "LOG: ${LOG}"
echo "--- key lines ---"
grep -E "step:500/|step:1000/|step:6[0-9]{3}/|stopping_early|final_sliding_window_exact|gptq:" \
    "${LOG}" | tail -20 || true

# Stack parity check
step500=$(grep "step:500/" "${LOG}" | grep -oP 'step_avg:\K[0-9.]+' || true)
if [[ -n "${step500}" ]]; then
    echo ""
    echo "step_avg @ 500: ${step500}ms  (target ~91ms, abort if >=93ms)"
    awk "BEGIN {exit !(${step500} >= 93.0)}" && true || {
        echo "STACK PARITY FAILURE — ${step500}ms >= 93ms. Wrong env. Score is invalid."
        exit 2
    }
fi
