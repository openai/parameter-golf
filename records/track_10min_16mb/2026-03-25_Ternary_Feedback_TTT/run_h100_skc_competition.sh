#!/bin/bash
# ============================================================================
# 8xH100 SXM - strict SKC competition run
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${DIR}/train_gpt.py" ]]; then
    TRAINER_PATH="${DIR}/train_gpt.py"
elif [[ -f "$(cd "${DIR}/../../.." 2>/dev/null && pwd)/train_gpt.py" ]]; then
    TRAINER_PATH="$(cd "${DIR}/../../.." && pwd)/train_gpt.py"
else
    TRAINER_PATH="./train_gpt.py"
fi
cd "${DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
FAST_SMOKE="${FAST_SMOKE:-0}"
OMP_THREADS="${OMP_NUM_THREADS:-1}"

# ── Data & tokenizer: competition regime ─────────────────────────────────────
export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_8192_bpe.model}"
export VOCAB_SIZE=8192

[[ -f "${TRAINER_PATH}" ]] || { echo "ERROR: train_gpt.py not found: ${TRAINER_PATH}" >&2; exit 1; }
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: TOKENIZER_PATH not found: ${TOKENIZER_PATH}" >&2; exit 1; }

# ── Architecture: opinionated SKC competition profile ────────────────────────
export ARCHITECTURE=skc
export NUM_LAYERS="${NUM_LAYERS:-11}"
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-4}"
export EMBED_DIM="${EMBED_DIM:-256}"
export PARTIAL_ROPE_DIMS="${PARTIAL_ROPE_DIMS:-32}"

export SKC_BLOCK_SIZE="${SKC_BLOCK_SIZE:-64}"
export SKC_NUM_CAPSULES="${SKC_NUM_CAPSULES:-16}"
export SKC_CAPSULE_DIM="${SKC_CAPSULE_DIM:-64}"
export SKC_CONV_KERNEL="${SKC_CONV_KERNEL:-4}"
export SKC_PARALLEL_RESIDUAL=1
export QK_GAIN_INIT="${QK_GAIN_INIT:-5.25}"

# Strip nonessential auxiliary machinery from the competition branch.
export FEEDBACK_ENABLED=0
export CAPSULE_ENABLED=0
export VRL_ENABLED=0
export KOOPMAN_ENABLED=0
export KOOPMAN_SPECULATOR_ENABLED=0
export MOE_ENABLED=0
export SHARED_BLOCKS="${SHARED_BLOCKS:-4}"
export WEIGHT_SHARING=0
export BIGRAM_HASH_ENABLED="${BIGRAM_HASH_ENABLED:-0}"
export BIGRAM_HASH_BUCKETS="${BIGRAM_HASH_BUCKETS:-3072}"
export BIGRAM_HASH_DIM="${BIGRAM_HASH_DIM:-112}"
export ENGRAM_NUM_HEADS="${ENGRAM_NUM_HEADS:-4}"
export ENGRAM_NUM_ORDERS="${ENGRAM_NUM_ORDERS:-2}"
export ENGRAM_INJECT_LAYER="${ENGRAM_INJECT_LAYER:-1}"
export XSA_START_LAYER=-1
export ADAPTIVE_HALT_ENABLED=0
export STOCHASTIC_DEPTH_PROB=0
export SELF_DISTILL_KL_WEIGHT=0

# ── Budget / throughput posture ──────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-570}"
export ITERATIONS=500000
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export COMPILER_WARMUP_STEPS="${COMPILER_WARMUP_STEPS:-20}"
export SEED="${SEED:-42}"
export COMPILE_MODE="${COMPILE_MODE:-max-autotune}"

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-262144}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export CURRICULUM_ENABLED=0

# ── Virtual depth ────────────────────────────────────────────────────────────
export TRAINING_DEPTH_RECURRENCE=0
export RECURRENCE_DEPTH="${RECURRENCE_DEPTH:-2}"
export RECURRENCE_START_FRACTION="${RECURRENCE_START_FRACTION:-0.35}"
export RECURRENCE_LAYERS="${RECURRENCE_LAYERS:-2,3,4,5}"
export EVAL_DEPTH_RECURRENCE="${EVAL_DEPTH_RECURRENCE:-2}"

# ── Optimizer: competition operating point ───────────────────────────────────
export MATRIX_OPTIMIZER=muon
export MATRIX_LR="${MATRIX_LR:-0.022}"
export SCALAR_LR="${SCALAR_LR:-0.001}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.025}"
export HEAD_LR="${HEAD_LR:-0.015}"
export MUON_WD="${MUON_WD:-0.095}"
export ADAM_WD="${ADAM_WD:-0.095}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-0}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.3}"
export WARMDOWN_FRACTION="${WARMDOWN_FRACTION:-0.72}"

# ── Weight averaging / evaluation hygiene ────────────────────────────────────
export EMA_ENABLED=1
export EMA_DECAY="${EMA_DECAY:-0.997}"
export EMA_START_FRACTION="${EMA_START_FRACTION:-0.20}"
export NGRAM_CACHE_ENABLED=0
export TEMP_SCALING=0
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE="${SLIDING_EVAL_STRIDE:-64}"
export EVAL_FEEDBACK_PASSES=0

# ── Legal score-first TTT ────────────────────────────────────────────────────
export TTT_ENABLED=1
export TTT_SCOPE="${TTT_SCOPE:-skc_safe}"
export TTT_LR="${TTT_LR:-0.005}"
export TTT_EPOCHS="${TTT_EPOCHS:-3}"
export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
export TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"
export TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-32}"
export TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"

# ── Ternary-native competition export ────────────────────────────────────────
export BITNET_GROUP_SIZE="${BITNET_GROUP_SIZE:-128}"
export FP_STORAGE="${FP_STORAGE:-fp4}"
export TURBO_QUANT_TRAIN=1
export TURBO_QUANT_EXPORT=1
export EXPORT_MODE=competition_ternary
export TERNARY_COMPRESS_BROTLI=1
export TERNARY_CLIP_MODE="${TERNARY_CLIP_MODE:-row_std}"
export TERNARY_CLIP_ROWS_K="${TERNARY_CLIP_ROWS_K:-12.85}"
export TERNARY_EMBED_CLIP_ROWS_K="${TERNARY_EMBED_CLIP_ROWS_K:-20.0}"
export EXPORT_ALIGNED_TRAIN=1
export EXPORT_ALIGNED_TRAIN_START_FRACTION="${EXPORT_ALIGNED_TRAIN_START_FRACTION:-0.80}"
export TERNARY_THRESHOLD_SEARCH=1
export TERNARY_THRESHOLD_LOW="${TERNARY_THRESHOLD_LOW:-0.02}"
export TERNARY_THRESHOLD_HIGH="${TERNARY_THRESHOLD_HIGH:-0.15}"
export TERNARY_THRESHOLD_STEPS="${TERNARY_THRESHOLD_STEPS:-4}"
export TERNARY_SCALE_SEARCH=1
export TERNARY_SCALE_MULT_LOW="${TERNARY_SCALE_MULT_LOW:-0.9}"
export TERNARY_SCALE_MULT_HIGH="${TERNARY_SCALE_MULT_HIGH:-1.1}"
export TERNARY_SCALE_MULT_STEPS="${TERNARY_SCALE_MULT_STEPS:-3}"
export TERNARY_CALIB_TOP_N="${TERNARY_CALIB_TOP_N:-5}"
export CALIB_MAX_CANDIDATES="${CALIB_MAX_CANDIDATES:-12}"
export CALIB_MAX_EVALS="${CALIB_MAX_EVALS:-32}"
export CALIB_MAX_SECONDS="${CALIB_MAX_SECONDS:-30}"
export EXPORT_PROXY_EVAL=1
export EXPORT_PROXY_EVERY="${EXPORT_PROXY_EVERY:-1200}"
export EXPORT_PROXY_NUM_SEQS="${EXPORT_PROXY_NUM_SEQS:-4}"
export EXPORT_PROXY_USE_BEST=1
export LZMA_PRESET="${LZMA_PRESET:-3}"
export GPTQ_LITE_ENABLED=0

# ── NCCL / smoke-test overrides ──────────────────────────────────────────────
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

if [[ "${FAST_SMOKE}" == "1" ]]; then
    export MAX_WALLCLOCK_SECONDS=45
    export COMPILE_MODE=none
    export TTT_ENABLED=0
    export EXPORT_PROXY_EVAL=0
    export EXPORT_ALIGNED_TRAIN=0
    export NGRAM_CACHE_ENABLED=0
    export TEMP_SCALING=0
fi

# ── Run ID ───────────────────────────────────────────────────────────────────
export RUN_ID="skc_h100_s${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
rm -f final_model.ternary.ptz final_model.competition.ptz submission.json pre_export_state.pt best_export_proxy_state.pt best_live_state.pt

echo "=========================================================================="
echo "  Strict SKC Competition Run - 8xH100 SXM"
echo "  RUN_ID : ${RUN_ID}"
echo "  MODEL  : arch=${ARCHITECTURE} L=${NUM_LAYERS} D=${MODEL_DIM} H=${NUM_HEADS} vocab=${VOCAB_SIZE} shared=${SHARED_BLOCKS}"
echo "  SKC    : block=${SKC_BLOCK_SIZE} caps=${SKC_NUM_CAPSULES} cap_dim=${SKC_CAPSULE_DIM} parallel=${SKC_PARALLEL_RESIDUAL}"
echo "  RECUR  : depth=${RECURRENCE_DEPTH} start=${RECURRENCE_START_FRACTION} layers=${RECURRENCE_LAYERS}"
echo "  BATCH  : tokens=${TRAIN_BATCH_TOKENS} seq=${TRAIN_SEQ_LEN} nproc=${NPROC_PER_NODE}"
echo "  OPT    : qk=${QK_GAIN_INIT} matrix_lr=${MATRIX_LR} scalar_lr=${SCALAR_LR} wd=${MUON_WD} warmdown=${WARMDOWN_FRACTION}"
echo "  HASH   : enabled=${BIGRAM_HASH_ENABLED} buckets=${BIGRAM_HASH_BUCKETS} dim=${BIGRAM_HASH_DIM} orders=${ENGRAM_NUM_ORDERS}"
echo "  TTT    : enabled=${TTT_ENABLED} scope=${TTT_SCOPE} lr=${TTT_LR} epochs=${TTT_EPOCHS} chunk=${TTT_CHUNK_TOKENS}"
echo "  EXPORT : mode=${EXPORT_MODE} clip=${TERNARY_CLIP_MODE} fp_storage=${FP_STORAGE} brotli=${TERNARY_COMPRESS_BROTLI} aligned=${EXPORT_ALIGNED_TRAIN}"
echo "  DATA   : ${DATA_PATH}"
echo "  TOKEN  : ${TOKENIZER_PATH}"
echo "  BUDGET : ${MAX_WALLCLOCK_SECONDS}s  smoke=${FAST_SMOKE}"
echo "=========================================================================="

LOG="${DIR}/logs/${RUN_ID}.log"
TORCHRUN_ENV=("OMP_NUM_THREADS=${OMP_THREADS}" "TORCH_NCCL_TIMEOUT_SEC=7200")
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    TORCHRUN_ENV+=("PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}")
fi

env "${TORCHRUN_ENV[@]}" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAINER_PATH}" 2>&1 | tee "${LOG}"

cp final_model.ternary.ptz "logs/${RUN_ID}_model.ternary.ptz" 2>/dev/null || true
cp final_model.competition.ptz "logs/${RUN_ID}_model.competition.ptz" 2>/dev/null || true
cp submission.json "logs/${RUN_ID}_submission.json" 2>/dev/null || true
cp pre_export_state.pt "logs/${RUN_ID}_pre_export_state.pt" 2>/dev/null || true
cp best_export_proxy_state.pt "logs/${RUN_ID}_best_export_proxy_state.pt" 2>/dev/null || true
cp best_live_state.pt "logs/${RUN_ID}_best_live_state.pt" 2>/dev/null || true

echo "=== DONE ==="
echo "Log      : ${LOG}"
echo "Artifact : logs/${RUN_ID}_model.ternary.ptz"
