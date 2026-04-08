#!/bin/bash
# ============================================================================
# 8×H100 SXM — hardware-aware SKC competition run
# ============================================================================
# This starts from the simple small-SKC line that converged best on 1x GPU, but
# it auto-scales model width and global batch on large multi-GPU hardware so we
# do not strand 8x H100s on a tiny per-rank workload.
# ============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
FAST_SMOKE="${FAST_SMOKE:-0}"
OMP_THREADS="${OMP_NUM_THREADS:-1}"
HW_AWARE_AUTO="${HW_AWARE_AUTO:-1}"
THROUGHPUT_FIRST="${THROUGHPUT_FIRST:-1}"
FORCE_MULTI_GPU="${FORCE_MULTI_GPU:-0}"
LAUNCH_NPROC_PER_NODE="${NPROC_PER_NODE}"

if command -v nvidia-smi >/dev/null 2>&1; then
    MIN_GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{m=$1} $1<m{m=$1} END{if(m!="") print m}')
else
    MIN_GPU_MEM_MIB=""
fi
if [[ -n "${MIN_GPU_MEM_MIB}" ]]; then
    MIN_GPU_MEM_GB=$(( (MIN_GPU_MEM_MIB + 1023) / 1024 ))
else
    MIN_GPU_MEM_GB=80
fi

USER_SET_TRAIN_BATCH_TOKENS=0
[[ -n "${TRAIN_BATCH_TOKENS+x}" ]] && USER_SET_TRAIN_BATCH_TOKENS=1
USER_SET_VAL_BATCH_SIZE=0
[[ -n "${VAL_BATCH_SIZE+x}" ]] && USER_SET_VAL_BATCH_SIZE=1

# ── Data (sp1024 — competition standard, vocab=1024) ─────────────────────────
export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024

[[ -f "${DIR}/train_gpt.py" ]] || { echo "ERROR: ${DIR}/train_gpt.py not found" >&2; exit 1; }
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: TOKENIZER_PATH not found: ${TOKENIZER_PATH}" >&2; exit 1; }

# ── Architecture: small SKC core ─────────────────────────────────────────────
export ARCHITECTURE=skc
HARDWARE_PROFILE="small_skc"
NUM_LAYERS_DEFAULT=8
MODEL_DIM_DEFAULT=256
NUM_HEADS_DEFAULT=4
NUM_KV_HEADS_DEFAULT=2
MLP_MULT_DEFAULT=4
SKC_BLOCK_SIZE_DEFAULT=16
SKC_NUM_CAPSULES_DEFAULT=8
SKC_CAPSULE_DIM_DEFAULT=64
SKC_CONV_KERNEL_DEFAULT=4
BIGRAM_HASH_DIM_DEFAULT=112
TRAIN_BATCH_TOKENS_DEFAULT=8192
VAL_BATCH_SIZE_DEFAULT=32768
MIN_PER_RANK_TOKENS=2048

if [[ "$HW_AWARE_AUTO" == "1" ]]; then
    if [[ "$THROUGHPUT_FIRST" != "1" ]] && (( NPROC_PER_NODE >= 8 && MIN_GPU_MEM_GB >= 70 )); then
        HARDWARE_PROFILE="h100_8x_scaleup"
        NUM_LAYERS_DEFAULT=8
        MODEL_DIM_DEFAULT=512
        NUM_HEADS_DEFAULT=8
        NUM_KV_HEADS_DEFAULT=4
        SKC_NUM_CAPSULES_DEFAULT=16
        SKC_CAPSULE_DIM_DEFAULT=128
        BIGRAM_HASH_DIM_DEFAULT=128
        TRAIN_BATCH_TOKENS_DEFAULT=32768
        VAL_BATCH_SIZE_DEFAULT=65536
        MIN_PER_RANK_TOKENS=4096
    elif [[ "$THROUGHPUT_FIRST" != "1" ]] && (( NPROC_PER_NODE >= 4 && MIN_GPU_MEM_GB >= 40 )); then
        HARDWARE_PROFILE="mid_multigpu_scaleup"
        NUM_LAYERS_DEFAULT=8
        MODEL_DIM_DEFAULT=384
        NUM_HEADS_DEFAULT=6
        NUM_KV_HEADS_DEFAULT=3
        SKC_NUM_CAPSULES_DEFAULT=12
        SKC_CAPSULE_DIM_DEFAULT=96
        BIGRAM_HASH_DIM_DEFAULT=128
        TRAIN_BATCH_TOKENS_DEFAULT=16384
        VAL_BATCH_SIZE_DEFAULT=49152
        MIN_PER_RANK_TOKENS=2048
    fi
fi

export NUM_LAYERS="${NUM_LAYERS:-$NUM_LAYERS_DEFAULT}"
export MODEL_DIM="${MODEL_DIM:-$MODEL_DIM_DEFAULT}"
export NUM_HEADS="${NUM_HEADS:-$NUM_HEADS_DEFAULT}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-$NUM_KV_HEADS_DEFAULT}"
export MLP_MULT="${MLP_MULT:-$MLP_MULT_DEFAULT}"

export SKC_BLOCK_SIZE="${SKC_BLOCK_SIZE:-$SKC_BLOCK_SIZE_DEFAULT}"
export SKC_NUM_CAPSULES="${SKC_NUM_CAPSULES:-$SKC_NUM_CAPSULES_DEFAULT}"
export SKC_CAPSULE_DIM="${SKC_CAPSULE_DIM:-$SKC_CAPSULE_DIM_DEFAULT}"
export SKC_CONV_KERNEL="${SKC_CONV_KERNEL:-$SKC_CONV_KERNEL_DEFAULT}"

# Keep the model path brutally simple. Fancy side modules have not earned their
# keep under the 10-minute budget.
export FEEDBACK_ENABLED=0
export CAPSULE_ENABLED=0
export VRL_ENABLED=0
export KOOPMAN_ENABLED=0
export KOOPMAN_SPECULATOR_ENABLED=0
export TTT_ENABLED=0
export EMA_ENABLED=0
export MOE_ENABLED=0
export SHARED_BLOCKS=0
export WEIGHT_SHARING=0
export XSA_START_LAYER=999

export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_BUCKETS=3072
export BIGRAM_HASH_DIM="${BIGRAM_HASH_DIM:-$BIGRAM_HASH_DIM_DEFAULT}"
export ENGRAM_NUM_HEADS=4
export ENGRAM_NUM_ORDERS=3
export ENGRAM_INJECT_LAYER=1
export PARTIAL_ROPE_DIMS=16
export LN_SCALE_DAMPING=1
export SMEARGATE_ENABLED=1

# ── Training budget ──────────────────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-599}
export ITERATIONS=500000
export WARMUP_STEPS=5
export COMPILER_WARMUP_STEPS=0
export SEED=${SEED:-42}
export COMPILE_MODE="${COMPILE_MODE:-none}"
export STOCHASTIC_DEPTH_PROB=0

# Match the winning small-model curriculum: 64 -> 128 -> 256.
export CURRICULUM_ENABLED=1
export CURRICULUM_PHASE1_FRAC=0.35
export CURRICULUM_PHASE2_FRAC=0.65
export CURRICULUM_PHASE3_FRAC=1.0
export CURRICULUM_PHASE4_FRAC=1.0
export CURRICULUM_PHASE5_FRAC=1.0
export CURRICULUM_PHASE1_SEQ=64
export CURRICULUM_PHASE2_SEQ=128
if [[ $USER_SET_TRAIN_BATCH_TOKENS -eq 0 ]]; then
    export TRAIN_BATCH_TOKENS="$TRAIN_BATCH_TOKENS_DEFAULT"
else
    export TRAIN_BATCH_TOKENS
fi
export TRAIN_SEQ_LEN=256
if [[ $USER_SET_VAL_BATCH_SIZE -eq 0 ]]; then
    export VAL_BATCH_SIZE="$VAL_BATCH_SIZE_DEFAULT"
else
    export VAL_BATCH_SIZE
fi
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"

PER_RANK_TOKENS=$(( TRAIN_BATCH_TOKENS / NPROC_PER_NODE ))
if [[ "$HW_AWARE_AUTO" == "1" && $USER_SET_TRAIN_BATCH_TOKENS -eq 0 && $PER_RANK_TOKENS -lt $MIN_PER_RANK_TOKENS ]]; then
    export TRAIN_BATCH_TOKENS=$(( MIN_PER_RANK_TOKENS * NPROC_PER_NODE ))
    PER_RANK_TOKENS=$(( TRAIN_BATCH_TOKENS / NPROC_PER_NODE ))
fi

LAUNCH_REASON="requested"
if [[ "$THROUGHPUT_FIRST" == "1" && "$FORCE_MULTI_GPU" != "1" && "$NPROC_PER_NODE" -gt 1 ]]; then
    if (( MODEL_DIM <= 256 && NUM_LAYERS <= 8 && TRAIN_BATCH_TOKENS / NPROC_PER_NODE <= 2048 )); then
        LAUNCH_NPROC_PER_NODE=1
        LAUNCH_REASON="tiny-model throughput-first collapse"
    fi
fi
LAUNCH_PER_RANK_TOKENS=$(( TRAIN_BATCH_TOKENS / LAUNCH_NPROC_PER_NODE ))

# ── Optimizer (low LR proven optimal for SKC from ablations) ─────────────────
export MATRIX_LR=0.02
export SCALAR_LR=0.015
export TIED_EMBED_LR=0.035
export MUON_WD=0.04
export ADAM_WD=0.04
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_BACKEND_STEPS=5
export GRAD_CLIP_NORM=0.3
export WARMDOWN_FRACTION=0.5
export TERNARY_NOISE_SCALE=0.0

# ── Weight averaging ─────────────────────────────────────────────────────────
export LAWA_ENABLED=1
export LAWA_K=5
export LAWA_FREQ=100
export SWA_ENABLED=1
export SWA_EVERY=50
export SWA_START_SCALE=0.2
export AVERAGE_TERNARY_PARAMS=0
export TKO_ENABLED=0
export SELF_DISTILL_KL_WEIGHT=0

# ── N-gram cache (free BPB at eval) ──────────────────────────────────────────
export NGRAM_CACHE_ENABLED=1
export NGRAM_MAX_ORDER=5
export NGRAM_ALPHA_BASE=0.05
export NGRAM_ALPHA_SCALE=0.55
export NGRAM_ENTROPY_CENTER=4.0

# ── Disabled (proven to hurt or neutral) ─────────────────────────────────────
# ── Quantization / export ────────────────────────────────────────────────────
export BITNET_GROUP_SIZE=128
export TURBO_QUANT_EXPORT=1
export TURBO_QUANT_TRAIN=0
export EXPORT_ALIGNED_TRAIN=0
export GPTQ_LITE_ENABLED=0
export HESSIAN_TERNARY_GPTQ=0
export SELECTIVE_PRUNING=0
export SAVE_PRE_EXPORT_STATE=1
export EXPORT_PROXY_EVAL="${EXPORT_PROXY_EVAL:-1}"
export EXPORT_PROXY_EVERY="${EXPORT_PROXY_EVERY:-1}"
export EXPORT_PROXY_NUM_SEQS="${EXPORT_PROXY_NUM_SEQS:-8}"
export EXPORT_PROXY_USE_BEST="${EXPORT_PROXY_USE_BEST:-1}"
export LOCAL_SGD_SYNC_EVERY="${LOCAL_SGD_SYNC_EVERY:-8}"   # LocalSGD: skip allreduce, average weights every N steps
export LOCAL_SGD_WARMUP_STEPS="${LOCAL_SGD_WARMUP_STEPS:-30}"

# ── Eval ─────────────────────────────────────────────────────────────────────
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE=32
export TEMP_SCALING=0

if [[ "$FAST_SMOKE" == "1" ]]; then
    export HESSIAN_TERNARY_GPTQ=0
    export GPTQ_LITE_ENABLED=0
    export SELECTIVE_PRUNING=0
    export SLIDING_EVAL=0
    export TEMP_SCALING=0
    export NGRAM_CACHE_ENABLED=0
    export SAVE_PRE_EXPORT_STATE=0
    export EXPORT_PROXY_EVAL=0
fi

# ── Run ID ───────────────────────────────────────────────────────────────────
export RUN_ID="skc_h100_s${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
rm -f final_model.ternary.ptz submission.json pre_export_state.pt best_export_proxy_state.pt best_live_state.pt
GPTQ_DESC="adaptive rotated/plain ternary export; no Hessian GPTQ; no pruning"
if [[ "$FAST_SMOKE" == "1" ]]; then
    GPTQ_DESC="disabled for smoke test"
fi

echo "=========================================================================="
echo "  Hardware-Aware SKC Competition Run — 8×H100 SXM"
echo "  RUN_ID : ${RUN_ID}"
echo "  HW     : profile=${HARDWARE_PROFILE}  min_gpu_mem=${MIN_GPU_MEM_GB}GB  hw_auto=${HW_AWARE_AUTO}"
echo "  MODEL  : SKC  L=${NUM_LAYERS}  D=${MODEL_DIM}  vocab=${VOCAB_SIZE}"
echo "  SKC    : block=${SKC_BLOCK_SIZE}  caps=${SKC_NUM_CAPSULES}  cap_dim=${SKC_CAPSULE_DIM}"
echo "  ENGRAM : buckets=${BIGRAM_HASH_BUCKETS}×${BIGRAM_HASH_DIM}  orders=${ENGRAM_NUM_ORDERS}"
echo "  CURR   : 64 -> 128 -> 256"
echo "  BATCH  : ${TRAIN_BATCH_TOKENS} tok/step  seq=${TRAIN_SEQ_LEN}  per_requested_rank=${PER_RANK_TOKENS}"
echo "  LAUNCH : nproc=${LAUNCH_NPROC_PER_NODE}/${NPROC_PER_NODE}  per_active_rank=${LAUNCH_PER_RANK_TOKENS}  reason=${LAUNCH_REASON}"
echo "  GPTQ   : ${GPTQ_DESC}"
echo "  BUDGET : ${MAX_WALLCLOCK_SECONDS}s"
echo "  SEED   : ${SEED}"
echo "  GPUS   : visible=${NPROC_PER_NODE} active=${LAUNCH_NPROC_PER_NODE}"
echo "  SMOKE  : ${FAST_SMOKE}"
echo "  DATA   : ${DATA_PATH}"
echo "=========================================================================="

LOG="${DIR}/logs/${RUN_ID}.log"
TORCHRUN_ENV=("OMP_NUM_THREADS=${OMP_THREADS}")
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    TORCHRUN_ENV+=("PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}")
fi
env "${TORCHRUN_ENV[@]}" \
    torchrun --standalone --nproc_per_node="${LAUNCH_NPROC_PER_NODE}" train_gpt.py 2>&1 | tee "$LOG"

# train_gpt.py writes final_model.ternary.ptz in CWD (/workspace)
cp final_model.ternary.ptz "logs/${RUN_ID}_model.ternary.ptz" 2>/dev/null || true
cp final_model.ternary.ptz "logs/mlx_reasoner_model.ternary.ptz" 2>/dev/null || true
cp submission.json "logs/${RUN_ID}_submission.json" 2>/dev/null || true
cp pre_export_state.pt "logs/${RUN_ID}_pre_export_state.pt" 2>/dev/null || true
cp best_export_proxy_state.pt "logs/${RUN_ID}_best_export_proxy_state.pt" 2>/dev/null || true
cp best_live_state.pt "logs/${RUN_ID}_best_live_state.pt" 2>/dev/null || true

echo "=== DONE ==="
echo "Log      : $LOG"
echo "Artifact : logs/${RUN_ID}_model.ternary.ptz"
