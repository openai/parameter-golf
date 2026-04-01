#!/bin/bash
set -euo pipefail
# ================================================================
#  BW8_Tap — Production Run
#
#  BW5 + CRAWLER_TAP_DIM=32 CRAWLER_TAP_LOOP_SPECIFIC=0
#  (shared encoder tap, all flat encoder layers, dim=32)
#
#  Gate evidence: −0.00352 int6_sw vs control (BW7 MegaGate, 2000 steps)
#  Target: beat 1.18672385 BPB (BW5 champion, seed=444)
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-01_BW8_Tap/run.sh
#    SEED=300 NPROC_PER_NODE=8 bash crawler/2026-04-01_BW8_Tap/run.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/BW8Tap_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

TORCHRUN=$(command -v torchrun 2>/dev/null || echo "python3 -m torch.distributed.run")

# ----------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || { echo "  ERROR: zstandard not found — pip install zstandard"; exit 1; }

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    import flash_attn; v=flash_attn.__version__
    if v.startswith('3'): print(f'  FA3 v{v} OK')
    else: print(f'  WARNING: FA{v[0]} — want FA3')
" 2>/dev/null || { echo "  ERROR: no flash_attn found — install before production run"; exit 1; }

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
echo "============================================"
echo "  BW8_Tap — Shared Encoder Tap dim=32"
echo "  BW5 + TAP_DIM=32 TAP_LOOP_SPECIFIC=0"
echo "  seed=${SEED}  GPUs=${NPROC}  wallclock=600s"
echo "  Log: ${LOG}"
echo "============================================"
echo ""

env \
    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS=600 \
    WARMDOWN_ITERS=2000 \
    COMPLEMENT_ALPHA=0 \
    XSA_LAST_N=11 \
    BIGRAM_VOCAB_SIZE=2048 \
    ROPE_DIMS=16 \
    SWA_EVERY=50 \
    MTP_NUM_HEADS=0 \
    LATE_QAT_THRESHOLD=0 \
    MATRIX_LR=0.03 \
    TORCHDYNAMO_OPTIMIZE_DDP=0 \
    COMPILE_FULLGRAPH=1 \
    NGRAM_EVAL_ORDER=0 \
    MODEL_DIM=512 \
    USE_CRAWLER=1 \
    NUM_FLAT_LAYERS=4 \
    NUM_CRAWLER_LAYERS=1 \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=6.0 \
    INST_DIM=32 \
    CRAWLER_QUANT_INT8=1 \
    DELTA_NET_HEADS=0 \
    SKIP_EMA=1 \
    SKIP_GPTQ=1 \
    LOOP_AWARE_GPTQ=0 \
    NITRUST_ENABLE=0 \
    NITRUST_STRICT=0 \
    MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_CHOKE_DIM=0 \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
    CRAWLER_LOOP_SMEAR=0 \
    CRAWLER_TAP_DIM=32 \
    CRAWLER_TAP_LOOP_SPECIFIC=0 \
    CRAWLER_TAP_LAYERS=all \
    NPROC_PER_NODE="${NPROC}" \
    ${TORCHRUN} --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
    2>&1 | tee "${LOG}"

# ----------------------------------------------------------------
# Extract and print summary
# ----------------------------------------------------------------
int6_bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
steps=$(grep -oP 'stopping_early.*step:\K[0-9]+' "${LOG}" | tail -1 \
      || grep -oP 'step:\K[0-9]+/20000 val_loss' "${LOG}" | tail -1 \
      || echo "?")
bytes=$(grep -oP 'Total submission size int6\+zstd: \K[0-9]+' "${LOG}" | tail -1 || echo "?")
step_ms=$(grep -oP 'step_avg\s+\K[0-9.]+' "${LOG}" | tail -1 || echo "?")

echo ""
echo "============================================"
echo "  RESULT — BW8_Tap seed=${SEED}"
echo "  steps:       ${steps}"
echo "  raw_bpb:     ${raw_bpb}"
echo "  int6_sw_bpb: ${int6_bpb}"
echo "  step_avg:    ${step_ms}ms  (target ~74.68ms)"
echo "  bytes:       ${bytes}  (limit 16000000)"
echo "  log:         ${LOG}"
echo ""
echo "  Champion:    1.18672385 BPB (BW5)"
echo "============================================"

# ----------------------------------------------------------------
# Auto-save checkpoint
# ----------------------------------------------------------------
CKPT_DIR="${REPO_ROOT}/checkpoints"
mkdir -p "${CKPT_DIR}"
CKPT_NAME="BW8Tap_s${SEED}_$(date +%Y%m%d_%H%M%S)_bpb${int6_bpb}.pt"
if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
    cp "${REPO_ROOT}/final_model.pt" "${CKPT_DIR}/${CKPT_NAME}"
    echo "  checkpoint: ${CKPT_DIR}/${CKPT_NAME}"
else
    echo "  WARNING: final_model.pt not found — checkpoint not saved"
fi
