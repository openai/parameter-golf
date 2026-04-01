#!/bin/bash
set -euo pipefail
# ================================================================
#  BW11_5Flat — 4×GPU gate (2000 steps)
#
#  Base: BW8 config (BW5 + TAP_DIM=32 shared)
#  Variable: NUM_FLAT_LAYERS=5 (5F+1C vs baseline 4F+1C)
#
#  Arms:
#    BWFF-00  BW8 control (NUM_FLAT_LAYERS=4)
#    BWFF-01  BW8 + NUM_FLAT_LAYERS=5
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-01_BW11_5Flat/gate.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

TORCHRUN=$(command -v torchrun 2>/dev/null || echo "python3 -m torch.distributed.run")

# Preflight: warn on missing FA3 (SDPA fallback active)
echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    try:
        import flash_attn; v=flash_attn.__version__
        if v.startswith('3'): print(f'  FA3 v{v} OK')
        else: print(f'  WARNING: FA{v[0]} — using SDPA fallback')
    except ImportError:
        print('  WARNING: no flash_attn — using SDPA fallback')
" 2>/dev/null || echo "  WARNING: flash_attn check failed — using SDPA fallback"

# BW8 base env (BW5 + TAP shared dim=32)
BASE_ENV=(
    SEED="${SEED}"
    ITERATIONS=2000
    MAX_WALLCLOCK_SECONDS=3600
    WARMDOWN_ITERS=2000
    COMPLEMENT_ALPHA=0
    XSA_LAST_N=11
    BIGRAM_VOCAB_SIZE=2048
    ROPE_DIMS=16
    SWA_EVERY=50
    MTP_NUM_HEADS=0
    LATE_QAT_THRESHOLD=0
    MATRIX_LR=0.03
    TORCHDYNAMO_OPTIMIZE_DDP=0
    COMPILE_FULLGRAPH=1
    NGRAM_EVAL_ORDER=0
    MODEL_DIM=512
    USE_CRAWLER=1
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=3
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8=1
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=32
    CRAWLER_TAP_LOOP_SPECIFIC=0
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    NPROC_PER_NODE="${NPROC}"
)

run_arm() {
    local label="$1"; shift
    local extra=("$@")
    local log="${RESULTS_DIR}/gate_${label}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "--- ARM ${label} ---"
    echo "Log: ${log}"
    env "${BASE_ENV[@]}" "${extra[@]}" \
        ${TORCHRUN} --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"
    raw=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || echo "?")
    int6=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || echo "?")
    step_ms=$(grep -oP 'step_avg\s+\K[0-9.]+' "${log}" | tail -1 || echo "?")
    bytes=$(grep -oP 'Total submission size int6\+zstd: \K[0-9]+' "${log}" | tail -1 || echo "?")
    echo "  ${label}: raw=${raw}  int6_sw=${int6}  step=${step_ms}ms  bytes=${bytes}"
}

echo "================================================================"
echo "  BW11_5Flat gate  seed=${SEED}  GPUs=${NPROC}"
echo "  Base: BW8 (TAP_DIM=32 shared)"
echo "  Variable: NUM_FLAT_LAYERS (4 vs 5)"
echo "================================================================"

run_arm "BWFF-00" NUM_FLAT_LAYERS=4
run_arm "BWFF-01" NUM_FLAT_LAYERS=5

echo ""
echo "================================================================"
echo "  SUMMARY"
echo "  Pass: BWFF-01 int6_sw < BWFF-00  AND  step_avg within 5ms"
echo "  Fail: regression or size exceeds 16MB"
echo "================================================================"
