#!/usr/bin/env bash
# ============================================================
# RASCAL BASELINE — STRICT MODE
# NO FALLBACK. NO ITERATIONS. FAIL FAST OR DO NOT RUN.
#
# Root causes of prior wasted runs (DO NOT RE-INTRODUCE):
#  1. Wrong GPTQ lane: SKIP_GPTQ was 0 → always 1 here, hardcoded
#  2. Wrong CUDA stack: cu130 gave ~93ms/step vs cu124's ~91ms/step
#
# Pre-flight checks happen BEFORE any compute.
# Any failure exits non-zero immediately.
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ── LOCKED CONSTANTS ─────────────────────────────────────────
EXPECTED_HASH="7b5bffe2601ff2fa54829a0b5b5dff7a5ad51894f2ea5a923a952c1477c7bfc6"
TRAIN_SCRIPT="analysis/pr1120_racecar_lab/copies/train_gpt_rascal_sota_local.py"
REQUIRED_CUDA_PREFIX="12.4"
STEP_AVG_WARN_MS=91.5   # post-warmup target from records/
STEP_AVG_ABORT_MS=93.0  # anything at or above this = wrong stack, abort
CHECK_STEP=500           # read step_avg from this step in log
LOG_DIR="analysis/pr1120_racecar_lab/runs_baseline_strict"

# ── USER PARAMS (only these are settable) ────────────────────
SEED="${SEED:-444}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# ── HELPERS ──────────────────────────────────────────────────
fail() { echo ""; echo "=== FAIL: $* ===" >&2; echo "DO NOT PROCEED." >&2; exit 1; }
ok()   { echo "  OK  $*"; }
warn() { echo "  WARN $*"; }

echo ""
echo "======================================================"
echo "  RASCAL BASELINE STRICT — PRE-FLIGHT"
echo "  seed=${SEED}  nproc=${NPROC_PER_NODE}"
echo "======================================================"

# ── CHECK 1: Source file exists ───────────────────────────────
echo "[1/4] Source file..."
[[ -f "${TRAIN_SCRIPT}" ]] || fail "Source not found: ${TRAIN_SCRIPT}"
ok "${TRAIN_SCRIPT}"

# ── CHECK 2: Source hash ──────────────────────────────────────
echo "[2/4] Source hash..."
actual_hash=$(sha256sum "${TRAIN_SCRIPT}" | awk '{print $1}')
if [[ "${actual_hash}" != "${EXPECTED_HASH}" ]]; then
  fail "Hash mismatch.
  expected: ${EXPECTED_HASH}
  actual:   ${actual_hash}
  The source has been modified. Restore from records/ before running."
fi
ok "hash match: ${actual_hash:0:16}..."

# ── CHECK 3: CUDA version must be 12.4.x (not cu130) ──────────
echo "[3/4] CUDA version (must be ${REQUIRED_CUDA_PREFIX}.x, not cu130)..."
cuda_ver=$(python3 -c "import torch; v=torch.version.cuda; print(v if v else 'NONE')" 2>/dev/null) \
  || fail "python3/torch import failed. Environment is broken."
torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "${cuda_ver}" != "${REQUIRED_CUDA_PREFIX}"* ]]; then
  fail "Wrong CUDA build: '${cuda_ver}' (torch ${torch_ver}).
  Required: ${REQUIRED_CUDA_PREFIX}.x
  This is the root cause of wasted run #2 (cu130 → 92.9–93ms/step).
  Fix environment: install torch+cu124, then re-run this script."
fi
ok "CUDA ${cuda_ver} | torch ${torch_ver}"

# ── CHECK 4: SKIP_GPTQ is hardcoded — confirm no env override ─
echo "[4/4] GPTQ lane lock (SKIP_GPTQ=1 hardcoded)..."
if [[ "${SKIP_GPTQ:-1}" != "1" ]]; then
  fail "Caller set SKIP_GPTQ=${SKIP_GPTQ}. This script locks to baseline lane (SKIP_GPTQ=1). Unset it."
fi
ok "SKIP_GPTQ=1 (baseline lane, naive int6)"

echo ""
echo "======================================================"
echo "  PRE-FLIGHT PASSED — LAUNCHING RUN"
echo "======================================================"
echo "  script:  ${TRAIN_SCRIPT}"
echo "  seed:    ${SEED}"
echo "  nproc:   ${NPROC_PER_NODE}"
echo "  cuda:    ${cuda_ver}"
echo "  torch:   ${torch_ver}"
echo "  target:  step_avg ~${STEP_AVG_WARN_MS}ms  abort_threshold: ${STEP_AVG_ABORT_MS}ms"
echo "======================================================"
echo ""

mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/baseline_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

# ── LAUNCH ────────────────────────────────────────────────────
env \
  SEED="${SEED}" \
  MAX_WALLCLOCK_SECONDS=600 \
  SKIP_GPTQ=1 \
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
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" \
  2>&1 | tee "${LOG}"

run_exit=${PIPESTATUS[0]}

# ── POST-RUN: Stack parity check ──────────────────────────────
echo ""
echo "[post-run] Stack parity check (step_avg at step ${CHECK_STEP})..."
step_avg_line=$(grep "step:${CHECK_STEP}/" "${LOG}" | head -n 1 || true)
if [[ -z "${step_avg_line}" ]]; then
  warn "step ${CHECK_STEP} not found in log — run may have stopped early."
else
  step_avg=$(echo "${step_avg_line}" | grep -oP 'step_avg:\K[0-9.]+' || true)
  if [[ -n "${step_avg}" ]]; then
    echo "  step_avg @ step ${CHECK_STEP}: ${step_avg}ms"
    if awk "BEGIN {exit !(${step_avg} >= ${STEP_AVG_ABORT_MS})}"; then
      echo ""
      echo "=== STACK PARITY FAILURE ==="
      echo "  step_avg ${step_avg}ms >= abort threshold ${STEP_AVG_ABORT_MS}ms"
      echo "  This matches the cu130 symptom (wasted run #2)."
      echo "  Score from this run is INVALID. Do not record."
      echo "  Fix: verify CUDA ${REQUIRED_CUDA_PREFIX}.x and re-run."
      exit 3
    elif awk "BEGIN {exit !(${step_avg} > ${STEP_AVG_WARN_MS})}"; then
      warn "step_avg ${step_avg}ms is above target ${STEP_AVG_WARN_MS}ms but below abort. Investigate."
    else
      ok "step_avg ${step_avg}ms — stack parity confirmed."
    fi
  fi
fi

# ── GPTQ line verification ────────────────────────────────────
echo "[post-run] GPTQ lane verification..."
gptq_line=$(grep -E "gptq:(SKIPPED|calibrated)" "${LOG}" | head -n 1 || true)
if [[ -z "${gptq_line}" ]]; then
  warn "No GPTQ status line found. Inspect log manually."
elif echo "${gptq_line}" | grep -q "calibrated"; then
  echo ""
  echo "=== WRONG LANE ==="
  echo "  GPTQ ran (calibrated) but SKIP_GPTQ=1 was set."
  echo "  This should not be possible. Inspect source for env override."
  exit 4
else
  ok "gptq:SKIPPED confirmed (baseline lane)"
fi

echo ""
echo "======================================================"
echo "  RUN COMPLETE"
echo "  log: ${LOG}"
echo "  exit: ${run_exit}"
echo "======================================================"
