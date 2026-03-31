#!/bin/bash
set -euo pipefail
# ================================================================
#  BW7 MegaGate — 8-arm sequential ablation
#
#  One variable per arm vs BW5 baseline. 2000 steps each.
#  Target pod: 4×H100.  NPROC=4  (~$0.25/arm, ~$2 total)
#
#  Arms:
#    CTRL-00  baseline (all defaults, BW5 config)
#    SMEAR-01 CRAWLER_LOOP_SMEAR=1
#    TAP-02   CRAWLER_TAP_DIM=32  CRAWLER_TAP_LOOP_SPECIFIC=1  (per-loop)
#    TAP-03   CRAWLER_TAP_DIM=32  CRAWLER_TAP_LOOP_SPECIFIC=0  (shared)
#    TAP-04   CRAWLER_TAP_DIM=16  CRAWLER_TAP_LOOP_SPECIFIC=1  (smaller)
#    ANC-05   ANCHOR_DIM=32
#    ANC-06   ANCHOR_DIM=64
#    FLAT-07  FLAT_WEIGHT_SHARE=1
#
#  Usage (from pod after git pull):
#    NPROC_PER_NODE=4 bash crawler/2026-03-31_BW7_MegaGate/run_ablation.sh
#    SEED=300 NPROC_PER_NODE=4 bash crawler/2026-03-31_BW7_MegaGate/run_ablation.sh
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

SUMMARY="${RESULTS_DIR}/summary_s${SEED}_$(date +%Y%m%d_%H%M%S).txt"

# ----------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || echo "  WARNING: zstandard not found"

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    try:
        import flash_attn; v=flash_attn.__version__
        if v.startswith('3'): print(f'  FA3 v{v} OK')
        else: print(f'  WARNING: FA{v[0]} — want FA3, will use SDPA fallback')
    except ImportError:
        print('  WARNING: no flash_attn — using PyTorch SDPA fallback (slower but correct)')
" 2>/dev/null || echo "  WARNING: flash_attn check failed — using PyTorch SDPA fallback"

echo "[preflight] checking data..."
python3 -c "
import glob
shards = glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin')
print(f'  train shards: {len(shards)}')
assert len(shards) >= 4, f'need >=4 shards, got {len(shards)}'
" || { echo "  ERROR: insufficient data shards"; exit 1; }

echo "[preflight] checking tokenizer..."
[[ -f "./data/tokenizers/fineweb_1024_bpe.model" ]] \
    || { echo "  ERROR: tokenizer not found"; exit 1; }
echo "  tokenizer OK"

echo ""
echo "================================================================"
echo "  BW7 MegaGate — 8 arms  seed=${SEED}  GPUs=${NPROC}"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""

# Write summary header
{
echo "BW7 MegaGate — seed=${SEED}  GPUs=${NPROC}  $(date)"
echo "Champion baseline: 1.18672385 BPB (BW5)"
echo ""
printf "%-10s  %-12s  %-12s  %-10s  %-10s  %s\n" \
    "ARM" "raw_bpb" "int6_sw_bpb" "step_ms" "bytes" "delta_vs_ctrl"
echo "----------  ------------  ------------  ----------  ----------  -------------"
} | tee "${SUMMARY}"

# ----------------------------------------------------------------
# Base env vars — BW5 config, unchanged for all arms
# ----------------------------------------------------------------
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
    NUM_FLAT_LAYERS=4
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=3
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8=1
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    NITRUST_ENABLE=0
    NITRUST_STRICT=0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=1
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    NPROC_PER_NODE="${NPROC}"
)

# ----------------------------------------------------------------
# run_arm <arm_id> <label> [EXTRA_VAR=val ...]
# ----------------------------------------------------------------
CTRL_RAW="?"
CTRL_INT6="?"

run_arm() {
    local arm_id="$1"
    local label="$2"
    shift 2
    local extra_env=("$@")

    local log="${RESULTS_DIR}/${arm_id}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  ARM ${arm_id}: ${label}  [${#extra_env[@]} overrides]"
    echo "  Log: ${log}"
    echo "----------------------------------------------------------------"

    env "${BASE_ENV[@]}" "${extra_env[@]}" \
        torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"

    local raw int6 step_ms bytes
    raw=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || echo "?")
    int6=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || echo "?")
    step_ms=$(grep -oP 'step_avg\s+\K[0-9.]+' "${log}" | tail -1 || echo "?")
    bytes=$(grep -oP 'Total submission size int6\+zstd: \K[0-9]+' "${log}" | tail -1 || echo "?")

    # Store control values
    if [[ "${arm_id}" == "CTRL-00" ]]; then
        CTRL_RAW="${raw}"
        CTRL_INT6="${int6}"
    fi

    # Compute delta vs control
    local delta="?"
    if [[ "${int6}" != "?" && "${CTRL_INT6}" != "?" ]]; then
        delta=$(python3 -c "
v = float('${int6}') - float('${CTRL_INT6}')
sign = '+' if v >= 0 else ''
print(f'{sign}{v:.5f}')
" 2>/dev/null || echo "?")
    fi

    printf "%-10s  %-12s  %-12s  %-10s  %-10s  %s\n" \
        "${arm_id}" "${raw}" "${int6}" "${step_ms}" "${bytes}" "${delta}" \
        | tee -a "${SUMMARY}"

    # Save checkpoint
    local ckpt_dir="${REPO_ROOT}/checkpoints"
    mkdir -p "${ckpt_dir}"
    local ckpt_name="BW7_${arm_id}_s${SEED}_$(date +%Y%m%d_%H%M%S)_bpb${int6}.pt"
    if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
        cp "${REPO_ROOT}/final_model.pt" "${ckpt_dir}/${ckpt_name}"
        echo "  checkpoint: ${ckpt_dir}/${ckpt_name}"
    else
        echo "  WARNING: final_model.pt not found — checkpoint not saved"
    fi
}

# ----------------------------------------------------------------
# Run all arms
# ----------------------------------------------------------------

run_arm "CTRL-00" "baseline (BW5 config)"

run_arm "SMEAR-01" "LoopSmearGate" \
    CRAWLER_LOOP_SMEAR=1

run_arm "TAP-02" "Tap dim=32 per-loop" \
    CRAWLER_TAP_DIM=32 \
    CRAWLER_TAP_LOOP_SPECIFIC=1 \
    CRAWLER_TAP_LAYERS=all

run_arm "TAP-03" "Tap dim=32 shared" \
    CRAWLER_TAP_DIM=32 \
    CRAWLER_TAP_LOOP_SPECIFIC=0 \
    CRAWLER_TAP_LAYERS=all

run_arm "TAP-04" "Tap dim=16 per-loop" \
    CRAWLER_TAP_DIM=16 \
    CRAWLER_TAP_LOOP_SPECIFIC=1 \
    CRAWLER_TAP_LAYERS=all

run_arm "ANC-05" "DeltaAnchor dim=32" \
    ANCHOR_DIM=32

run_arm "ANC-06" "DeltaAnchor dim=64" \
    ANCHOR_DIM=64

run_arm "FLAT-07" "SharedFlatWeights" \
    FLAT_WEIGHT_SHARE=1

# ----------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  BW7 MegaGate COMPLETE — seed=${SEED}"
echo "  Champion to beat: 1.18672385 BPB (BW5)"
echo "================================================================"
cat "${SUMMARY}"
echo ""
echo "Full logs: ${RESULTS_DIR}/"
