#!/usr/bin/env bash
# ============================================================================
# Single-GPU CUDA SKC proof run
# Goal: show the current CUDA SKC stack converges well on a cheap RunPod GPU.
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

OMP_THREADS="${OMP_NUM_THREADS:-1}"
FAST_SMOKE="${FAST_SMOKE:-0}"
PROOF_MODE="${PROOF_MODE:-1}"

export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024

[[ -f "${TRAINER_PATH}" ]] || { echo "ERROR: ${DIR}/train_gpt.py not found" >&2; exit 1; }
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: TOKENIZER_PATH not found: ${TOKENIZER_PATH}" >&2; exit 1; }

# ── Core SKC config ──────────────────────────────────────────────────────────
export ARCHITECTURE=skc
export NUM_LAYERS=8
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export SKC_BLOCK_SIZE=16
export SKC_NUM_CAPSULES=16
export SKC_CAPSULE_DIM=128
export SKC_CONV_KERNEL=4

# ── Locked convergence features from the CUDA/H100 path ─────────────────────
export XSA_START_LAYER=0
export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_BUCKETS=3072
export BIGRAM_HASH_DIM=112
export ENGRAM_NUM_HEADS=4
export ENGRAM_NUM_ORDERS=3
export ENGRAM_INJECT_LAYER=1
export PARTIAL_ROPE_DIMS=16
export LN_SCALE_DAMPING=1
export CURRICULUM_ENABLED=1
export CURRICULUM_PHASE1_FRAC=0.015625
export CURRICULUM_PHASE2_FRAC=0.046875
export CURRICULUM_PHASE3_FRAC=0.109375
export CURRICULUM_PHASE4_FRAC=0.234375
export CURRICULUM_PHASE5_FRAC=0.484375
export CURRICULUM_PHASE1_SEQ=64
export CURRICULUM_PHASE2_SEQ=128
export CURRICULUM_PHASE3_SEQ=256
export CURRICULUM_PHASE4_SEQ=512
export CURRICULUM_PHASE5_SEQ=1024

# ── Budget and batch for a cheap single GPU ─────────────────────────────────
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1800}"
export ITERATIONS=500000
export WARMUP_STEPS=5
export SEED="${SEED:-1337}"
export COMPILE_MODE="${COMPILE_MODE:-none}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-32768}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32768}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"

# ── Optimizer ────────────────────────────────────────────────────────────────
export MATRIX_LR=0.02
export SCALAR_LR=0.015
export TIED_EMBED_LR=0.025
export MUON_WD=0.04
export ADAM_WD=0.04
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=0
export MUON_BACKEND_STEPS=5
export GRAD_CLIP_NORM=0.3
export WARMDOWN_FRACTION="${WARMDOWN_FRACTION:-0.5}"
export STOCHASTIC_DEPTH_PROB=0

# ── Weight averaging / proven toggles ───────────────────────────────────────
export LAWA_ENABLED="${LAWA_ENABLED:-0}"
export LAWA_K=10
export LAWA_FREQ=100
export SWA_ENABLED="${SWA_ENABLED:-1}"
export SWA_EVERY=50
export AVERAGE_TERNARY_PARAMS="${AVERAGE_TERNARY_PARAMS:-0}"
export SMEARGATE_ENABLED=1
export TKO_ENABLED=0
export SELF_DISTILL_KL_WEIGHT="${SELF_DISTILL_KL_WEIGHT:-0}"

# ── Disabled features ────────────────────────────────────────────────────────
export CAPSULE_ENABLED=0
export FEEDBACK_ENABLED=0
export KOOPMAN_SPECULATOR_ENABLED=0
export VRL_ENABLED=0
export TTT_ENABLED=0
export EMA_ENABLED=0
export MOE_ENABLED=0
export SHARED_BLOCKS=0

# ── Serialization / eval ────────────────────────────────────────────────────
export BITNET_GROUP_SIZE=128
export TURBO_QUANT_EXPORT=1
export TURBO_QUANT_TRAIN=0
export GPTQ_LITE_ENABLED="${GPTQ_LITE_ENABLED:-0}"
export HESSIAN_TERNARY_GPTQ="${HESSIAN_TERNARY_GPTQ:-0}"
export SELECTIVE_PRUNING="${SELECTIVE_PRUNING:-0}"
export SLIDING_EVAL="${SLIDING_EVAL:-0}"
export SLIDING_EVAL_STRIDE=64
export TEMP_SCALING="${TEMP_SCALING:-0}"
export NGRAM_CACHE_ENABLED="${NGRAM_CACHE_ENABLED:-0}"
export NGRAM_MAX_ORDER=5
export NGRAM_ALPHA_BASE=0.05
export NGRAM_ALPHA_SCALE=0.55
export NGRAM_ENTROPY_CENTER=4.0

if [[ "$FAST_SMOKE" == "1" ]]; then
    export MAX_WALLCLOCK_SECONDS=45
    export VAL_LOSS_EVERY=0
    export TRAIN_LOG_EVERY=1
    export GPTQ_LITE_ENABLED=0
    export HESSIAN_TERNARY_GPTQ=0
    export SELECTIVE_PRUNING=0
    export SLIDING_EVAL=0
    export TEMP_SCALING=0
    export NGRAM_CACHE_ENABLED=0
fi

export RUN_ID="cheap_skc_s${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

echo "=========================================================================="
echo "  Cheap-GPU SKC Proof Run"
echo "  RUN_ID : ${RUN_ID}"
echo "  MODEL  : SKC L=${NUM_LAYERS} D=${MODEL_DIM} heads=${NUM_HEADS}/${NUM_KV_HEADS}"
echo "  DATA   : ${DATA_PATH}"
echo "  BATCH  : ${TRAIN_BATCH_TOKENS} tok/step  seq=${TRAIN_SEQ_LEN}"
echo "  BUDGET : ${MAX_WALLCLOCK_SECONDS}s"
echo "  SEED   : ${SEED}"
echo "  LAWA   : ${LAWA_ENABLED}  SWA=${SWA_ENABLED}  SMEARGATE=${SMEARGATE_ENABLED}"
echo "  EXPORT : turbo=${TURBO_QUANT_EXPORT} gptq=${GPTQ_LITE_ENABLED} hess=${HESSIAN_TERNARY_GPTQ} prune=${SELECTIVE_PRUNING}"
echo "  EVAL   : sliding=${SLIDING_EVAL} ngram=${NGRAM_CACHE_ENABLED} temp=${TEMP_SCALING}"
echo "  SMOKE  : ${FAST_SMOKE}"
echo "=========================================================================="

LOG="${DIR}/logs/${RUN_ID}.log"
env OMP_NUM_THREADS="${OMP_THREADS}" \
    torchrun --standalone --nproc_per_node=1 "${TRAINER_PATH}" 2>&1 | tee "$LOG"

cp final_model.ternary.ptz "logs/${RUN_ID}_model.ternary.ptz" 2>/dev/null || true
cp submission.json "logs/${RUN_ID}_submission.json" 2>/dev/null || true

echo "=== DONE ==="
echo "Log      : $LOG"
echo "Artifact : logs/${RUN_ID}_model.ternary.ptz"
