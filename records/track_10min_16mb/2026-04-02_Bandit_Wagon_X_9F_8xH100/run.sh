#!/bin/bash
set -euo pipefail
# ================================================================
# Bandit Wagon X 9F — production full run (submission-oriented)
#
# One seed per invocation (default: 444). Use again with SEED=300.
# This script enforces submission-size legality by default.
#
# Usage:
#   SEED=444 NPROC_PER_NODE=8 bash records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/run.sh
#   SEED=300 NPROC_PER_NODE=8 bash records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/run.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
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
CRAWLER_LOOPS="${CRAWLER_LOOPS:-3}"
CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8:-0}"   # 0 keeps artifact size safer for 16MB cap

mkdir -p "${SCRIPT_DIR}/logs"
LOG_TS="${SCRIPT_DIR}/logs/train_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
LOG="${SCRIPT_DIR}/train_seed${SEED}.log"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

# ----------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || { echo "  ERROR: zstandard missing (pip install zstandard)"; exit 1; }

echo "[preflight] checking flash_attn..."
python3 - <<'PY'
try:
    import flash_attn_interface  # type: ignore
    print("  FA3 (hopper) OK")
except Exception:
    try:
        import flash_attn  # type: ignore
        v = flash_attn.__version__
        if str(v).startswith("3"):
            print(f"  FA3 v{v} OK")
        else:
            print(f"  WARNING: flash-attn v{v} detected (want v3)")
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
echo "  Bandit Wagon X 9F — full run"
echo "  seed=${SEED} GPUs=${NPROC} wallclock=${MAX_WALLCLOCK_SECONDS}s"
echo "  NUM_FLAT_LAYERS=${NUM_FLAT_LAYERS} NUM_CRAWLER_LAYERS=${NUM_CRAWLER_LAYERS} CRAWLER_LOOPS=${CRAWLER_LOOPS}"
echo "  CRAWLER_QUANT_INT8=${CRAWLER_QUANT_INT8}  (0=smaller artifacts, 1=higher risk for >16MB)"
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
    CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8}" \
    DELTA_NET_HEADS=0 \
    SKIP_EMA=1 \
    SKIP_GPTQ=1 \
    LOOP_AWARE_GPTQ=0 \
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
bytes_total="$(grep -oP 'Total submission size int6\+(?:zstd|zlib): \K[0-9]+' "${LOG}" | tail -1 || true)"
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
echo "  RESULT — Bandit Wagon X 9F seed=${SEED}"
echo "  model_params:  ${model_params:-?}"
echo "  raw_bpb:       ${raw_bpb:-?}"
echo "  int6_sw_bpb:   ${int6_sw_bpb:-?}"
echo "  step_avg_ms:   ${step_ms:-?}"
echo "  steps:         ${steps:-?}"
echo "  train_time_s:  ${train_time_s}"
echo "  bytes_total:   ${bytes_total:-?}  (limit ${LEGAL_SIZE_LIMIT})"
echo "  bytes_code:    ${code_bytes:-?}"
echo "  artifact_legal:${artifact_ok}"
echo "  log:           ${LOG}"
echo "============================================"

METRICS_TSV="${SCRIPT_DIR}/metrics_seed${SEED}.tsv"
{
    echo -e "seed\tmodel_params\traw_bpb\tint6_sw_bpb\tsteps\tstep_ms\ttrain_time_s\tbytes_total\tbytes_code\tartifact_legal\tlog"
    echo -e "${SEED}\t${model_params:-?}\t${raw_bpb:-?}\t${int6_sw_bpb:-?}\t${steps:-?}\t${step_ms:-?}\t${train_time_s}\t${bytes_total:-?}\t${code_bytes:-?}\t${artifact_ok}\t${LOG}"
} > "${METRICS_TSV}"

# Keep uniquely named artifacts for submission packaging.
if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
    cp -f "${REPO_ROOT}/final_model.pt" "${SCRIPT_DIR}/final_model_seed${SEED}.pt"
fi
if [[ -f "${REPO_ROOT}/final_model.int6.ptz" ]]; then
    cp -f "${REPO_ROOT}/final_model.int6.ptz" "${SCRIPT_DIR}/final_model_seed${SEED}.int6.ptz"
fi
if [[ -f "${REPO_ROOT}/final_model.int8.ptz" ]]; then
    cp -f "${REPO_ROOT}/final_model.int8.ptz" "${SCRIPT_DIR}/final_model_seed${SEED}.int8.ptz"
fi

if [[ "${ENFORCE_SIZE_LIMIT}" == "1" && "${artifact_ok}" == "no" ]]; then
    echo "ERROR: artifact exceeds ${LEGAL_SIZE_LIMIT} bytes. Re-run with smaller config or set ENFORCE_SIZE_LIMIT=0."
    exit 2
fi

exit 0
