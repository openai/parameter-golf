#!/bin/bash
# QK_GAIN_SLOT_Gate — single-GPU cross-correlation ablation
# Tests: QK_GAIN_INIT=4.0 (training-side) and SLOT (eval-side)
# 4 cases: baseline / qk_gain4 / slot_only / qk_gain4_slot
# ~1200 steps each, seed=444, COPRIME_MAX_LOADED_SHARDS=1
#
# BEFORE RUNNING: pod smoke-test runs first (10 steps).
# Abort if step_avg > 200ms — broken pod, reprovision.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC:-1}"
TORCHRUN="${TORCHRUN:-torchrun}"
CASES="${CASES:-all}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

# ── Preflight ────────────────────────────────────────────────────────────────
echo "[preflight] tokenizer: ${TOKENIZER_PATH}"
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: tokenizer not found"; exit 1; }
echo "[preflight] data: ${DATA_PATH}"
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: data path not found"; exit 1; }
python3 -c "import zstandard; print('[preflight] zstandard OK')" 2>/dev/null \
    || echo "[preflight] WARNING: zstandard not found"
python3 -c "
try:
    import flash_attn_interface; print('[preflight] FA3 (hopper) OK')
except ImportError:
    try:
        import flash_attn; v=flash_attn.__version__
        if v.startswith('3'): print(f'[preflight] FA3 v{v} OK')
        else: print(f'[preflight] WARNING: FA{v[0]} detected — want FA3')
    except ImportError:
        print('[preflight] WARNING: no flash_attn found')
" 2>/dev/null

# ── Smoke test ────────────────────────────────────────────────────────────────
if [[ "${SKIP_SMOKE}" == "0" ]]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "  SMOKE TEST (10 steps — checking step time)"
    echo "  Expected: ~91ms/step on H100 SXM"
    echo "  Abort threshold: >200ms/step = broken pod"
    echo "════════════════════════════════════════════"

    SMOKE_LOG="${SCRIPT_DIR}/logs/smoke_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "${SCRIPT_DIR}/logs"

    env ITERATIONS=10 \
        WARMDOWN_ITERS=0 \
        SKIP_FINAL_EVAL=1 \
        SKIP_GPTQ=1 \
        COMPILE_ENABLED=1 \
        COMPILE_FULLGRAPH=1 \
        COPRIME_MAX_LOADED_SHARDS=1 \
        COPRIME_SHARDS_PER_BATCH=1 \
        LOADER_MODE=coprime \
        TRAIN_BATCH_TOKENS=786432 \
        TRAIN_SEQ_LEN=2048 \
        TRAIN_LOG_EVERY=1 \
        MAX_WALLCLOCK_SECONDS=0 \
        VAL_LOSS_EVERY=99999 \
        SEED="${SEED}" \
        DATA_PATH="${DATA_PATH}" \
        TOKENIZER_PATH="${TOKENIZER_PATH}" \
        PYTHONPATH="${PYTHONPATH:-}" \
        "${TORCHRUN}" --standalone --nproc_per_node="${NPROC}" \
            "${SCRIPT_DIR}/train_gpt.py" 2>&1 | tee "${SMOKE_LOG}"

    # Parse step_avg from log
    STEP_AVG=$(grep -oP 'step_avg:\K[\d.]+' "${SMOKE_LOG}" | tail -1 || echo "")
    if [[ -z "${STEP_AVG}" ]]; then
        echo ""
        echo "ERROR: could not parse step_avg from smoke log. Check ${SMOKE_LOG}"
        exit 1
    fi

    echo ""
    echo "[smoke] step_avg: ${STEP_AVG}ms"

    # Threshold scales with GPU count:
    #   8xH100 → ~91ms/step (grad_accum=1)
    #   1xH100 → ~730ms/step (grad_accum=8, same total batch)
    # Formula: 91ms * (8 / NPROC) * 2.5 safety margin
    THRESHOLD=$(( 91 * 8 / NPROC * 5 / 2 ))
    STEP_INT="${STEP_AVG%%.*}"
    if [[ "${STEP_INT}" -gt "${THRESHOLD}" ]]; then
        echo "ABORT: step_avg=${STEP_AVG}ms exceeds ${THRESHOLD}ms threshold (nproc=${NPROC})."
        echo "This pod is broken (wrong GPU, throttling, or driver issue)."
        echo "Destroy and reprovision before spending money on ablations."
        exit 1
    fi

    echo "[smoke] PASSED (${STEP_AVG}ms/step) — pod is healthy"
    echo ""
fi

# ── Ablation ──────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════"
echo "  QK_GAIN_SLOT ABLATION"
echo "  Seed: ${SEED}  nproc: ${NPROC}"
echo "  Cases: ${CASES}"
echo "  1200 steps, COPRIME_MAX_LOADED_SHARDS=1"
echo "════════════════════════════════════════════"

python3 "${SCRIPT_DIR}/run_ablation.py" \
    --seed "${SEED}" \
    --nproc "${NPROC}" \
    --torchrun "${TORCHRUN}" \
    --cases ${CASES}

echo "════════════════════════════════════════════"
echo "  DONE — results in ${SCRIPT_DIR}/logs/"
echo "════════════════════════════════════════════"
