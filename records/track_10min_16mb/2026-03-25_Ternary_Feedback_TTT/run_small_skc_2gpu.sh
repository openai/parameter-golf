#!/usr/bin/env bash
# ============================================================================
# Cheap 2-GPU CUDA proxy for the local small-model SKC winner.
# Target: maximize throughput and preserve the local 8L/640D convergence recipe.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
# Auto-discover trainer path (local or project root)
if [[ -f "${SCRIPT_DIR:-.}/train_gpt.py" ]]; then
    TRAINER_PATH="${SCRIPT_DIR:-.}/train_gpt.py"
elif [[ -f "$(cd "${SCRIPT_DIR:-.}/../../.." 2>/dev/null && pwd)/train_gpt.py" ]]; then
    TRAINER_PATH="$(cd "${SCRIPT_DIR:-.}/../../.." && pwd)/train_gpt.py"
else
    # Fallback for scripts that don't define SCRIPT_DIR
    TRAINER_PATH="./train_gpt.py"
fi
cd "$DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"   # Default 2 — this is a 2-GPU script
OMP_THREADS="${OMP_NUM_THREADS:-1}"
FAST_SMOKE="${FAST_SMOKE:-0}"

export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024

[[ -f "${TRAINER_PATH}" ]] || { echo "ERROR: ${DIR}/train_gpt.py not found" >&2; exit 1; }
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: TOKENIZER_PATH not found: ${TOKENIZER_PATH}" >&2; exit 1; }

if command -v nvidia-smi >/dev/null 2>&1; then
    MIN_GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{m=$1} $1<m{m=$1} END{if(m!="") print m}')
else
    MIN_GPU_MEM_MIB=""
fi
if [[ -n "${MIN_GPU_MEM_MIB}" ]]; then
    MIN_GPU_MEM_GB=$(( (MIN_GPU_MEM_MIB + 1023) / 1024 ))
else
    MIN_GPU_MEM_GB=24
fi

# ── Hardware-optimal config for 2x A40 (<200ms @ seq=2048) ──────────────────
# D=640 = 5×128 → perfect tensor core tile alignment on A40
# H=8, head_dim=80 → optimal cuBLAS GEMM tile
# KV=2 → 4:1 GQA compression keeps KV cache lean
# 4 MoE experts, top-1 → max expressiveness within 16MB ternary budget
export ARCHITECTURE="skc"
export NUM_LAYERS="${NUM_LAYERS:-8}"
export MODEL_DIM="${MODEL_DIM:-640}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
export MLP_MULT="${MLP_MULT:-4}"

# MoE enabled and scaled to 4 experts (~14MB export size at D=640)
export MOE_ENABLED=1
export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
export MOE_TOP_K=1
export MOE_START_FRACTION=0.30
export MOE_ROUTER_AUX_LOSS_COEF=0.01

# Spectral Koopman Capsule
export SKC_NUM_CAPSULES="${SKC_NUM_CAPSULES:-24}"
export SKC_CAPSULE_DIM="${SKC_CAPSULE_DIM:-96}"
export SKC_CONV_KERNEL="${SKC_CONV_KERNEL:-4}"
export SKC_BLOCK_SIZE="${SKC_BLOCK_SIZE:-64}"

# Outer CapsuleBank and Feedback disabled (integrated inside SKC blocks instead)
export FEEDBACK_ENABLED=0
export CAPSULE_ENABLED=0
export VRL_ENABLED=1
export KOOPMAN_ENABLED=1
export KOOPMAN_RANK=8
export KOOPMAN_SPECULATOR_ENABLED=0
export TTT_ENABLED=0     # Disabled: expensive post-training eval path extends total wall-clock beyond 10min
export EMA_ENABLED=1

export XSA_START_LAYER=999

# ── Ternary-friendly extras that actually survived local ablations ───────────
export BIGRAM_HASH_ENABLED="${BIGRAM_HASH_ENABLED:-1}"
export BIGRAM_HASH_BUCKETS="${BIGRAM_HASH_BUCKETS:-16384}"
export BIGRAM_HASH_DIM="${BIGRAM_HASH_DIM:-112}"
export ENGRAM_NUM_HEADS="${ENGRAM_NUM_HEADS:-4}"
export ENGRAM_NUM_ORDERS="${ENGRAM_NUM_ORDERS:-3}"
export ENGRAM_INJECT_LAYER=1
export PARTIAL_ROPE_DIMS=16
export LN_SCALE_DAMPING=1

# ── Aggressive curriculum: fast ramp, ~80% of budget at full seq=2048 ────────
export CURRICULUM_ENABLED=1
# Fractions are wallclock-based. Ramp through short contexts quickly,
# land at full seq=2048 by 20% of budget (120s of 599s).
export CURRICULUM_PHASE1_FRAC=0.04   # 0–4%:   seq=64   (~24s)
export CURRICULUM_PHASE2_FRAC=0.08   # 4–8%:   seq=128  (~24s)
export CURRICULUM_PHASE3_FRAC=0.13   # 8–13%:  seq=256  (~30s)
export CURRICULUM_PHASE4_FRAC=0.18   # 13–18%: seq=512  (~30s)
export CURRICULUM_PHASE5_FRAC=0.24   # 18–24%: seq=1024 (~36s)
# 24–100%: seq=2048 ← 76% of budget at full context (~455s)
export CURRICULUM_PHASE1_SEQ=64
export CURRICULUM_PHASE2_SEQ=128
export CURRICULUM_PHASE3_SEQ=256
export CURRICULUM_PHASE4_SEQ=512
export CURRICULUM_PHASE5_SEQ=1024

# ── Training budget ──────────────────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-599}"
export ITERATIONS=500000
export GRAD_ACCUM_STEPS=1  # Force 1 — never inherit from parent env
export WARMUP_STEPS=5
export COMPILER_WARMUP_STEPS=0
export SEED="${SEED:-42}"
export COMPILE_MODE="${COMPILE_MODE:-none}"

# 4096 tokens/step = 1 sequence per GPU at seq=2048.
if [[ -z "${TRAIN_BATCH_TOKENS:-}" ]]; then
    export TRAIN_BATCH_TOKENS=32768
else
    export TRAIN_BATCH_TOKENS
fi
export TRAIN_SEQ_LEN=2048
if [[ -z "${VAL_BATCH_SIZE:-}" ]]; then
    if (( MIN_GPU_MEM_GB <= 12 )); then
        export VAL_BATCH_SIZE=8192
    elif (( MIN_GPU_MEM_GB <= 16 )); then
        export VAL_BATCH_SIZE=16384
    else
        export VAL_BATCH_SIZE=262144  # Maximize VRAM usage — note: capped at 131k tokens in trainer
    fi
else
    export VAL_BATCH_SIZE
fi
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"    # 0 = no mid-training val, maximizes training steps
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"

# Optimizer settings
export MATRIX_LR=0.005
export SCALAR_LR=0.001
export TIED_EMBED_LR=0.004
export MUON_WD=0.090
export ADAM_WD=0.090
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_BACKEND_STEPS=3
export GRAD_CLIP_NORM=0.3
export WARMDOWN_FRACTION=0.20

# Weight averaging: EMA is the only implemented averaging mechanism.
export EMA_ENABLED=1
export EMA_DECAY=0.997
export EMA_START_FRACTION=0.20

# ── Export/eval path ──────────────────────────────────────────────────────────
export BITNET_GROUP_SIZE=128
export TURBO_QUANT_TRAIN="${TURBO_QUANT_TRAIN:-1}"   # Must match EXPORT — Hadamard rotation applied at both train & export
export TURBO_QUANT_EXPORT="${TURBO_QUANT_EXPORT:-1}"
export EXPORT_ALIGNED_TRAIN="${EXPORT_ALIGNED_TRAIN:-0}"
export EXPORT_ALIGNED_TRAIN_START_FRACTION="${EXPORT_ALIGNED_TRAIN_START_FRACTION:-0.0}"
export TERNARY_THRESHOLD_SEARCH="${TERNARY_THRESHOLD_SEARCH:-0}"
export TERNARY_THRESHOLD_LOW="${TERNARY_THRESHOLD_LOW:-0.35}"
export TERNARY_THRESHOLD_HIGH="${TERNARY_THRESHOLD_HIGH:-0.65}"
export TERNARY_THRESHOLD_STEPS="${TERNARY_THRESHOLD_STEPS:-5}"
export TERNARY_SCALE_SEARCH="${TERNARY_SCALE_SEARCH:-0}"
export TERNARY_SCALE_MULT_LOW="${TERNARY_SCALE_MULT_LOW:-0.85}"
export TERNARY_SCALE_MULT_HIGH="${TERNARY_SCALE_MULT_HIGH:-1.15}"
export TERNARY_SCALE_MULT_STEPS="${TERNARY_SCALE_MULT_STEPS:-3}"
export TURBO_QUANT_KV="${TURBO_QUANT_KV:-1}"
export GPTQ_LITE_ENABLED="${GPTQ_LITE_ENABLED:-0}"
export SLIDING_EVAL="${SLIDING_EVAL:-0}"          # Off for proxy run — extends total wall-clock beyond 10min
export SLIDING_EVAL_STRIDE="${SLIDING_EVAL_STRIDE:-32}"
export TEMP_SCALING="${TEMP_SCALING:-0}"          # Off for proxy run — post-training overhead
export NGRAM_CACHE_ENABLED="${NGRAM_CACHE_ENABLED:-0}"  # Off for proxy run — expensive post-training computation
export LZMA_PRESET="${LZMA_PRESET:-4}"  # Reliable and fast for 16MB limit
export EXPORT_PROXY_EVAL="${EXPORT_PROXY_EVAL:-1}"  # ON: capture best checkpoint across MoE spikes
export EXPORT_PROXY_EVERY="${EXPORT_PROXY_EVERY:-1200}"
export EXPORT_PROXY_NUM_SEQS="${EXPORT_PROXY_NUM_SEQS:-4}"
export EXPORT_PROXY_USE_BEST="${EXPORT_PROXY_USE_BEST:-1}"
export NGRAM_MAX_ORDER="${NGRAM_MAX_ORDER:-5}"
export NGRAM_ALPHA_BASE="${NGRAM_ALPHA_BASE:-0.05}"
export NGRAM_ALPHA_SCALE="${NGRAM_ALPHA_SCALE:-0.55}"
export NGRAM_ENTROPY_CENTER="${NGRAM_ENTROPY_CENTER:-4.0}"

if [[ "$FAST_SMOKE" == "1" ]]; then
    export MAX_WALLCLOCK_SECONDS=10
    export ITERATIONS=2
    export VAL_LOSS_EVERY=0
    export TRAIN_LOG_EVERY=1
    export GPTQ_LITE_ENABLED=0
    export SLIDING_EVAL=0
    export TEMP_SCALING=0
    export NGRAM_CACHE_ENABLED=0
    export EXPORT_PROXY_EVAL=0
    export VAL_BATCH_SIZE=16384
    export TTT_ENABLED=0
fi

# NCCL: force SHM path on PCIe-connected GPUs (P2P probe hangs indefinitely on RunPod)
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

export RUN_ID="small_skc_2gpu_s${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
rm -f final_model.ternary.ptz submission.json

echo "=========================================================================="
echo "  Cheap Multi-GPU Small SKC Run"
echo "  RUN_ID : ${RUN_ID}"
echo "  MODEL  : SKC L=${NUM_LAYERS} D=${MODEL_DIM} H=${NUM_HEADS}/${NUM_KV_HEADS}"
echo "  SKC    : block=${SKC_BLOCK_SIZE} caps=${SKC_NUM_CAPSULES} cap_dim=${SKC_CAPSULE_DIM}"
echo "  ENGRAM : buckets=${BIGRAM_HASH_BUCKETS} dim=${BIGRAM_HASH_DIM} orders=${ENGRAM_NUM_ORDERS}"
echo "  TERNARY: train_align=${EXPORT_ALIGNED_TRAIN}@${EXPORT_ALIGNED_TRAIN_START_FRACTION} thr_search=${TERNARY_THRESHOLD_SEARCH} scale_search=${TERNARY_SCALE_SEARCH}"
echo "  CURR   : 64 -> 128 -> 256 -> 512 -> 1024 @ 24% / 76%"
echo "  BATCH  : ${TRAIN_BATCH_TOKENS} tok/step  seq=${TRAIN_SEQ_LEN}"
echo "  VRAM   : min_gpu_mem=${MIN_GPU_MEM_GB}GB  val_batch=${VAL_BATCH_SIZE}"
echo "  BUDGET : ${MAX_WALLCLOCK_SECONDS}s"
echo "  SEED   : ${SEED}"
echo "  GPUS   : ${NPROC_PER_NODE}"
echo "  EXPORT : turbo=${TURBO_QUANT_EXPORT} sliding=${SLIDING_EVAL} ngram=${NGRAM_CACHE_ENABLED}"
echo "  SMOKE  : ${FAST_SMOKE}"
echo "=========================================================================="

LOG="${DIR}/logs/${RUN_ID}.log"
env OMP_NUM_THREADS="${OMP_THREADS}" \
    python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py 2>&1 | tee "$LOG"

# Trainer writes: final_model.ternary.ptz, submission.json
cp final_model.ternary.ptz "logs/${RUN_ID}_model.ternary.ptz" 2>/dev/null || true
cp submission.json "logs/${RUN_ID}_submission.json" 2>/dev/null || true
cp final_model.ternary.ptz "logs/mlx_reasoner_model.ternary.ptz" 2>/dev/null || true

echo "=== DONE ==="
echo "Log      : $LOG"
echo "Artifact : logs/${RUN_ID}_model.ternary.ptz"
