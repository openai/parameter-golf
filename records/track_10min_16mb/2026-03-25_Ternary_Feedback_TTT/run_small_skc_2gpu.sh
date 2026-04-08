#!/usr/bin/env bash
# ============================================================================
# Cheap 2-GPU CUDA proxy for the local small-model SKC winner.
# Target: maximize throughput and preserve the local 8L/256 convergence recipe.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
OMP_THREADS="${OMP_NUM_THREADS:-1}"
FAST_SMOKE="${FAST_SMOKE:-0}"

export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024

[[ -f "${DIR}/train_gpt.py" ]] || { echo "ERROR: ${DIR}/train_gpt.py not found" >&2; exit 1; }
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
# D=384 = 3×128 → perfect tensor core tile alignment on A40 (128-wide tiles)
# H=6, head_dim=64 → optimal cuBLAS GEMM tile (64-wide K dim)
# KV=2 → 3:1 GQA compression keeps KV cache lean
# 16 MoE experts, top-2 → max expressiveness within 16MB ternary budget
export ARCHITECTURE="skc"
export NUM_LAYERS="${NUM_LAYERS:-8}"
export MODEL_DIM="${MODEL_DIM:-640}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
export MLP_MULT="${MLP_MULT:-2}"

# C9/C12: MoE enabled and scaled to 4 experts (~14MB export size at D=640)
export MOE_ENABLED=1
export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
export MOE_TOP_K=1
export MOE_START_FRACTION=0.65
export MOE_ROUTER_AUX_LOSS_COEF=0.02

# Spectral Koopman Capsule — scaled down to match D=384
export SKC_NUM_CAPSULES="${SKC_NUM_CAPSULES:-24}"
export SKC_CAPSULE_DIM="${SKC_CAPSULE_DIM:-96}"
export SKC_CONV_KERNEL="${SKC_CONV_KERNEL:-4}"
export SKC_BLOCK_SIZE="${SKC_BLOCK_SIZE:-64}"

# C9/C10: Outer CapsuleBank and Feedback disabled (integrated inside SKC blocks instead)
export FEEDBACK_ENABLED=0
export CAPSULE_ENABLED=0
export DEQ_FEEDBACK=0
export DEQ_MAX_ITER=3
export VRL_ENABLED=1
export KOOPMAN_ENABLED=1
export KOOPMAN_RANK=8
export KOOPMAN_SPECULATOR_ENABLED=0
export TTT_ENABLED=1
export EMA_ENABLED=1

# Advanced Sprint Options
export WEIGHT_SHARING=1
export INSIDE_OUT_TRAINING=1
# C3: TKO_ENABLED=1 removed — it was dead code (line 161 overrides it to 0)
export XSA_START_LAYER=999

# ── Ternary-friendly extras that actually survived local ablations ───────────
export BIGRAM_HASH_ENABLED="${BIGRAM_HASH_ENABLED:-1}"
export BIGRAM_HASH_BUCKETS="${BIGRAM_HASH_BUCKETS:-16384}"  # sweep winner used 16384
export BIGRAM_HASH_DIM="${BIGRAM_HASH_DIM:-112}"
export ENGRAM_NUM_HEADS="${ENGRAM_NUM_HEADS:-4}"
export ENGRAM_NUM_ORDERS="${ENGRAM_NUM_ORDERS:-3}"
export ENGRAM_INJECT_LAYER=1
export PARTIAL_ROPE_DIMS=16
export LN_SCALE_DAMPING=1
export SMEARGATE_ENABLED=1

# ── Aggressive curriculum: fast ramp, ~80% of budget at full seq=2048 ────────
export CURRICULUM_ENABLED=1
# Fractions are wallclock-based. Ramp through short contexts quickly,
# land at full seq=2048 by 20% of budget (120s of 599s).
export CURRICULUM_PHASE1_FRAC=0.05   # 0–5%:   seq=64   (~30s)
export CURRICULUM_PHASE2_FRAC=0.10   # 5–10%:  seq=128  (~30s)
export CURRICULUM_PHASE3_FRAC=0.17   # 10–17%: seq=256  (~42s)
export CURRICULUM_PHASE4_FRAC=0.25   # 17–25%: seq=512  (~48s)
export CURRICULUM_PHASE5_FRAC=0.35   # 25–35%: seq=1024 (~60s)
# 35–100%: seq=2048 ← 65% of budget at full context (~390s)
export CURRICULUM_PHASE1_SEQ=64
export CURRICULUM_PHASE2_SEQ=128
export CURRICULUM_PHASE3_SEQ=256
export CURRICULUM_PHASE4_SEQ=512
export CURRICULUM_PHASE5_SEQ=1024
export CURRICULUM_ADAPTIVE_BATCH=0

# ── Training budget ──────────────────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-599}"
export ITERATIONS=500000
export GRAD_ACCUM_STEPS=1  # Force 1 — never inherit from parent env
export WARMUP_STEPS=5
export COMPILER_WARMUP_STEPS=0
export SEED="${SEED:-42}"
export COMPILE_MODE="${COMPILE_MODE:-none}"
export TRAINING_DEPTH_RECURRENCE="${TRAINING_DEPTH_RECURRENCE:-0}"  # 0=off (maximize steps); set >0 for deeper effective network (costs step time)
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-0}"   # Enable with depth recurrence to save VRAM

# 4096 tokens/step = 1 sequence per GPU at seq=2048.
# This prevents the SIGABRT/OOM at full context on A40 GPUs.
if [[ -z "${TRAIN_BATCH_TOKENS:-}" ]]; then
    export TRAIN_BATCH_TOKENS=4096
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
        export VAL_BATCH_SIZE=262144  # Maximize VRAM usage for fast eval
    fi
else
    export VAL_BATCH_SIZE
fi
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"    # 0 = no mid-training val, maximizes training steps
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"

# C9: Reduced LR overrides — Phase B Python defaults already set these to safe values;
# shell was overriding them back up to 0.02–0.035 which was too aggressive for SKC.
export MATRIX_LR=0.005
export SCALAR_LR=0.001
export TIED_EMBED_LR=0.004
export MUON_WD=0.04
export ADAM_WD=0.04
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_BACKEND_STEPS=3
export GRAD_CLIP_NORM=0.3
export WARMDOWN_FRACTION=0.5
export STOCHASTIC_DEPTH_PROB=0
export TERNARY_NOISE_SCALE=0.0

# ── Weight averaging: keep local recipe, but rely on selective averaging fix ─
export LAWA_ENABLED=1
export LAWA_K=5
export LAWA_FREQ=100
export SWA_ENABLED=1
export SWA_EVERY=50
export SWA_START_SCALE=0.2
export AVERAGE_TERNARY_PARAMS=0
export TKO_ENABLED=0
export SELF_DISTILL_KL_WEIGHT=0

# ── Export/eval path: faithful to the local small-model line ─────────────────
export BITNET_GROUP_SIZE=128
export TURBO_QUANT_TRAIN="${TURBO_QUANT_TRAIN:-0}"
export TURBO_QUANT_EXPORT="${TURBO_QUANT_EXPORT:-1}"
export EXPORT_ALIGNED_TRAIN="${EXPORT_ALIGNED_TRAIN:-1}"
export EXPORT_ALIGNED_TRAIN_START_FRACTION="${EXPORT_ALIGNED_TRAIN_START_FRACTION:-0.85}"
export TERNARY_THRESHOLD_SEARCH="${TERNARY_THRESHOLD_SEARCH:-1}"  # Critical for good roundtrip BPB
export TERNARY_THRESHOLD_LOW="${TERNARY_THRESHOLD_LOW:-0.35}"
export TERNARY_THRESHOLD_HIGH="${TERNARY_THRESHOLD_HIGH:-0.65}"
export TERNARY_THRESHOLD_STEPS="${TERNARY_THRESHOLD_STEPS:-5}"
export TERNARY_SCALE_SEARCH="${TERNARY_SCALE_SEARCH:-1}"   # Critical for good roundtrip BPB
export TERNARY_SCALE_MULT_LOW="${TERNARY_SCALE_MULT_LOW:-0.85}"
export TERNARY_SCALE_MULT_HIGH="${TERNARY_SCALE_MULT_HIGH:-1.15}"
export TERNARY_SCALE_MULT_STEPS="${TERNARY_SCALE_MULT_STEPS:-3}"
export TURBO_QUANT_KV="${TURBO_QUANT_KV:-1}"
export GPTQ_LITE_ENABLED="${GPTQ_LITE_ENABLED:-0}"
export HESSIAN_TERNARY_GPTQ="${HESSIAN_TERNARY_GPTQ:-0}"
export SELECTIVE_PRUNING="${SELECTIVE_PRUNING:-0}"
export SLIDING_EVAL="${SLIDING_EVAL:-1}"
export SLIDING_EVAL_STRIDE="${SLIDING_EVAL_STRIDE:-32}"
export TEMP_SCALING="${TEMP_SCALING:-1}"
export NGRAM_CACHE_ENABLED="${NGRAM_CACHE_ENABLED:-1}"  # Off for dev — expensive post-training computation
export SAVE_PRE_EXPORT_STATE="${SAVE_PRE_EXPORT_STATE:-0}"  # Off for dev — saves time and disk
export FAST_EXPORT="${FAST_EXPORT:-0}"          # Skip variant grid — LAWA directly, fast export
export LZMA_PRESET="${LZMA_PRESET:-4}"  # Reliable and fast for 16MB limit
export EXPORT_PROXY_EVAL="${EXPORT_PROXY_EVAL:-0}"  # Off — avoid mid-training export proxy overhead
export EXPORT_PROXY_EVERY="${EXPORT_PROXY_EVERY:-1}"
export EXPORT_PROXY_NUM_SEQS="${EXPORT_PROXY_NUM_SEQS:-8}"
export EXPORT_PROXY_USE_BEST="${EXPORT_PROXY_USE_BEST:-1}"
export LOCAL_SGD_SYNC_EVERY="${LOCAL_SGD_SYNC_EVERY:-1}"   # Full DDP: sync every step for maximum convergence
export LOCAL_SGD_WARMUP_STEPS="${LOCAL_SGD_WARMUP_STEPS:-0}"  # Start syncing from step 0
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
    export HESSIAN_TERNARY_GPTQ=0
    export SELECTIVE_PRUNING=0
    export SLIDING_EVAL=0
    export TEMP_SCALING=0
    export NGRAM_CACHE_ENABLED=0
    export SAVE_PRE_EXPORT_STATE=0
    export EXPORT_PROXY_EVAL=0
    export FAST_EXPORT=1
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
rm -f final_model.ternary.ptz submission.json pre_export_state.pt best_export_proxy_state.pt best_live_state.pt

echo "=========================================================================="
echo "  Cheap Multi-GPU Small SKC Run"
echo "  RUN_ID : ${RUN_ID}"
echo "  MODEL  : SKC L=${NUM_LAYERS} D=${MODEL_DIM} H=${NUM_HEADS}/${NUM_KV_HEADS}"
echo "  SKC    : block=${SKC_BLOCK_SIZE} caps=${SKC_NUM_CAPSULES} cap_dim=${SKC_CAPSULE_DIM}"
echo "  ENGRAM : buckets=${BIGRAM_HASH_BUCKETS} dim=${BIGRAM_HASH_DIM} orders=${ENGRAM_NUM_ORDERS}"
echo "  TERNARY: train_align=${EXPORT_ALIGNED_TRAIN}@${EXPORT_ALIGNED_TRAIN_START_FRACTION} thr_search=${TERNARY_THRESHOLD_SEARCH} scale_search=${TERNARY_SCALE_SEARCH}"
echo "  CURR   : 64 -> 128 -> 256 @ 35% / 65%"
echo "  BATCH  : ${TRAIN_BATCH_TOKENS} tok/step  seq=${TRAIN_SEQ_LEN}"
echo "  VRAM   : min_gpu_mem=${MIN_GPU_MEM_GB}GB  val_batch=${VAL_BATCH_SIZE}"
echo "  BUDGET : ${MAX_WALLCLOCK_SECONDS}s"
echo "  SEED   : ${SEED}"
echo "  GPUS   : ${NPROC_PER_NODE}"
echo "  EXPORT : turbo=${TURBO_QUANT_EXPORT} hess=${HESSIAN_TERNARY_GPTQ} sliding=${SLIDING_EVAL} ngram=${NGRAM_CACHE_ENABLED}"
echo "  SMOKE  : ${FAST_SMOKE}"
echo "=========================================================================="

LOG="${DIR}/logs/${RUN_ID}.log"
env OMP_NUM_THREADS="${OMP_THREADS}" \
    python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py 2>&1 | tee "$LOG"

cp final_model.ternary.ptz "logs/${RUN_ID}_model.ternary.ptz" 2>/dev/null || true
cp submission.json "logs/${RUN_ID}_submission.json" 2>/dev/null || true
cp pre_export_state.pt "logs/${RUN_ID}_pre_export_state.pt" 2>/dev/null || true
cp best_export_proxy_state.pt "logs/${RUN_ID}_best_export_proxy_state.pt" 2>/dev/null || true
cp best_live_state.pt "logs/${RUN_ID}_best_live_state.pt" 2>/dev/null || true
cp final_model.ternary.ptz "logs/mlx_reasoner_model.ternary.ptz" 2>/dev/null || true

echo "=== DONE ==="
echo "Log      : $LOG"
echo "Artifact : logs/${RUN_ID}_model.ternary.ptz"
