#!/bin/bash
set -euo pipefail
# ================================================================
# Bandit Wagon XI — Best-foot-forward production run
#
# BWX 9F + brotli + loop-aware GPTQ + QK4 + loops=2
#
# Usage:
#   SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-02_BWXI_Brotli_GPTQ/run.sh
#   SEED=300 NPROC_PER_NODE=8 bash crawler/2026-04-02_BWXI_Brotli_GPTQ/run.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
LEGAL_SIZE_LIMIT="${LEGAL_SIZE_LIMIT:-16000000}"
ENFORCE_SIZE_LIMIT="${ENFORCE_SIZE_LIMIT:-1}"

MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
WARMDOWN_ITERS="${WARMDOWN_ITERS:-2000}"
NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-9}"
NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-1}"
CRAWLER_LOOPS="${CRAWLER_LOOPS:-2}"

mkdir -p "${SCRIPT_DIR}/results"
LOG_TS="${SCRIPT_DIR}/results/train_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
LOG="${SCRIPT_DIR}/train_seed${SEED}.log"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

# ----------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------
echo "[preflight] checking brotli..."
python3 -c "import brotli; print(f'  brotli OK')" 2>/dev/null \
    || { echo "  installing brotli..."; pip install brotli -q; }

echo "[preflight] checking flash_attn..."
python3 - <<'PY'
try:
    import flash_attn_interface
    print("  FA3 (hopper) OK")
except Exception:
    try:
        import flash_attn
        v = flash_attn.__version__
        print(f"  flash-attn v{v}")
    except Exception:
        raise SystemExit("  ERROR: flash-attn not importable")
PY

echo "[preflight] checking dataset + tokenizer..."
python3 - <<'PY'
import glob, os
tok = "./data/tokenizers/fineweb_1024_bpe.model"
assert os.path.isfile(tok), f"missing tokenizer: {tok}"
shards = glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
assert len(shards) >= 8, f"need >=8 train shards, found {len(shards)}"
print(f"  tokenizer OK, train shards={len(shards)}")
PY

echo ""
echo "============================================"
echo "  Bandit Wagon XI — Best-foot-forward"
echo "  seed=${SEED} GPUs=${NPROC} wallclock=${MAX_WALLCLOCK_SECONDS}s"
echo "  NUM_FLAT_LAYERS=${NUM_FLAT_LAYERS} CRAWLER_LOOPS=${CRAWLER_LOOPS}"
echo "  LOOP_AWARE_GPTQ=1  GPTQ_CAL_SAMPLES=128"
echo "  QK_GAIN_INIT=4.0  Compression: brotli (quality=11)"
echo "  log: ${LOG_TS}"
echo "============================================"
echo ""

env \
    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
    WARMDOWN_ITERS="${WARMDOWN_ITERS}" \
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
    NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS}" \
    NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS}" \
    CRAWLER_LOOPS="${CRAWLER_LOOPS}" \
    CRAWLER_MLP_MULT=6.0 \
    INST_DIM=32 \
    CRAWLER_QUANT_INT8=0 \
    DELTA_NET_HEADS=0 \
    SKIP_EMA=1 \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=1 \
    QK_GAIN_INIT=4.0 \
    GPTQ_CAL_SAMPLES=128 \
    GPTQ_CAL_SEQ_LEN=2048 \
    MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_CHOKE_DIM=0 \
    CRAWLER_MLP_CHOKE_SHAPE=flat \
    CRAWLER_MLP_CHOKE_GROUPS=8 \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
    CRAWLER_LOOP_SMEAR=0 \
    CRAWLER_TAP_DIM=0 \
    CRAWLER_TAP_LOOP_SPECIFIC=1 \
    CRAWLER_TAP_LAYERS=all \
    ANCHOR_DIM=0 \
    FLAT_WEIGHT_SHARE=0 \
    NPROC_PER_NODE="${NPROC}" \
    "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
    2>&1 | tee "${LOG_TS}"

cp -f "${LOG_TS}" "${LOG}"

# ----------------------------------------------------------------
# Metrics extraction
# ----------------------------------------------------------------
raw_bpb="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || true)"
int6_sw_bpb="$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || true)"
bytes_total="$(grep -oP 'Total submission size int6\+(?:brotli|zlib): \K[0-9]+' "${LOG}" | tail -1 || true)"
code_bytes="$(grep -oP 'Code size: \K[0-9]+' "${LOG}" | tail -1 || true)"
step_ms="$(grep -oP 'step_avg:\K[0-9.]+' "${LOG}" | tail -1 || true)"
model_params="$(grep -oP 'model_params:\K[0-9]+' "${LOG}" | tail -1 || true)"
steps="$(grep -oP 'stopping_early:.*step:\K[0-9]+' "${LOG}" | tail -1 || true)"
if [[ -z "${steps}" ]]; then
    steps="$(grep -oP 'step:\K[0-9]+(?=/[0-9]+ val_loss:)' "${LOG}" | tail -1 || true)"
fi
train_time_ms="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+ train_time:\K[0-9]+' "${LOG}" | tail -1 || true)"
if [[ -n "${train_time_ms}" ]]; then
    train_time_s=$((train_time_ms / 1000))
else
    train_time_s="${MAX_WALLCLOCK_SECONDS}"
fi
gptq_time="$(grep -oP 'gptq:calibrated [0-9]+ layers in \K[0-9.]+' "${LOG}" | tail -1 || true)"

artifact_ok="unknown"
if [[ -n "${bytes_total}" && "${bytes_total}" =~ ^[0-9]+$ ]]; then
    if (( bytes_total <= LEGAL_SIZE_LIMIT )); then
        artifact_ok="yes"
    else
        artifact_ok="no"
    fi
fi

echo ""
echo "============================================"
echo "  RESULT — Bandit Wagon XI seed=${SEED}"
echo "  model_params:  ${model_params:-?}"
echo "  raw_bpb:       ${raw_bpb:-?}"
echo "  int6_sw_bpb:   ${int6_sw_bpb:-?}"
echo "  step_avg_ms:   ${step_ms:-?}"
echo "  steps:         ${steps:-?}"
echo "  train_time_s:  ${train_time_s}"
echo "  bytes_total:   ${bytes_total:-?}  (limit ${LEGAL_SIZE_LIMIT})"
echo "  bytes_code:    ${code_bytes:-?}"
echo "  gptq_cal_s:    ${gptq_time:-?}"
echo "  artifact_legal:${artifact_ok}"
echo "  log:           ${LOG}"
echo "============================================"

# Keep uniquely named artifacts
for f in final_model.pt final_model.int6.ptz final_model.int8.ptz; do
    if [[ -f "${REPO_ROOT}/${f}" ]]; then
        base="${f%.*}"
        ext="${f##*.}"
        cp -f "${REPO_ROOT}/${f}" "${SCRIPT_DIR}/${base}_seed${SEED}.${ext}"
    fi
done

if [[ "${ENFORCE_SIZE_LIMIT}" == "1" && "${artifact_ok}" == "no" ]]; then
    echo "ERROR: artifact exceeds ${LEGAL_SIZE_LIMIT} bytes."
    exit 2
fi

exit 0
