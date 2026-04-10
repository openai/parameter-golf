#!/usr/bin/env bash
# ============================================================================
# Pod-side ablation runner — Final Architecture Sweep (Aligned)
# Usage:
#   MODE=validate RUN_TAG=mytag bash run_single_ablation.sh <CONFIG_NAME>
#   bash run_single_ablation.sh <CONFIG_NAME> [screen|validate] [run_tag]
# ============================================================================
set -euo pipefail

CONFIG="${1:-baseline}"
MODE="${MODE:-}"
RUN_TAG="${RUN_TAG:-}"
if [[ -z "$MODE" && $# -ge 2 ]]; then
    case "$2" in
        screen|validate) MODE="$2" ;;
        *) [[ -z "$RUN_TAG" ]] && RUN_TAG="$2" ;;
    esac
fi
if [[ -z "$RUN_TAG" && $# -ge 3 ]]; then
    RUN_TAG="$3"
fi
MODE="${MODE:-screen}"
FAST_SMOKE="${FAST_SMOKE:-0}"
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

# ── Architecture Defaults ───────────────────────────────────────────────────
export ARCHITECTURE=skc
export NUM_LAYERS=8
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export SKC_NUM_CAPSULES=16
export SKC_CAPSULE_DIM=128
export SKC_CONV_KERNEL=4
export LAWA_ENABLED="${LAWA_ENABLED:-0}"  # Defaulting to OFF based on export stability findings
export SWA_ENABLED="${SWA_ENABLED:-1}"
export SWA_EVERY="${SWA_EVERY:-50}"
export AVERAGE_TERNARY_PARAMS="${AVERAGE_TERNARY_PARAMS:-0}"
export TKO_ENABLED=0
export EMA_ENABLED=0
export EMA_EVAL_APPLY=0
export SELF_DISTILL_KL_WEIGHT="${SELF_DISTILL_KL_WEIGHT:-0.02}"
export SELF_DISTILL_START_FRACTION="${SELF_DISTILL_START_FRACTION:-0.10}"

# ── User-Locked Features ────────────────────────────────────────────────────
export SKC_BLOCK_SIZE=16
export BIGRAM_HASH_BUCKETS=3072
export BIGRAM_HASH_DIM=128
export BIGRAM_HASH_ENABLED=1
export ENGRAM_NUM_HEADS=4
export ENGRAM_NUM_ORDERS=3
export ENGRAM_INJECT_LAYER=1
export CURRICULUM_ENABLED=1
export CURRICULUM_PHASE1_FRAC=0.010000
export CURRICULUM_PHASE2_FRAC=0.030000
export CURRICULUM_PHASE3_FRAC=0.080000
export CURRICULUM_PHASE4_FRAC=0.160000
export CURRICULUM_PHASE5_FRAC=0.300000
export CURRICULUM_PHASE1_SEQ=64
export CURRICULUM_PHASE2_SEQ=128
export CURRICULUM_PHASE3_SEQ=256
export CURRICULUM_PHASE4_SEQ=512
export CURRICULUM_PHASE5_SEQ=1024
export XSA_START_LAYER=0
export PARTIAL_ROPE_DIMS=16
export LN_SCALE_DAMPING=1
export SMEARGATE_ENABLED=1
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
export COMPILE_MODE="${COMPILE_MODE:-none}"
export TURBO_QUANT_TRAIN="${TURBO_QUANT_TRAIN:-1}"
export TURBO_QUANT_TRAIN_START_FRACTION="${TURBO_QUANT_TRAIN_START_FRACTION:-0.20}"

# ── Training & Budget ────────────────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-420}"  # 7 min
export SEED="${SEED:-1337}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-32768}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32768}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
export COMPILER_WARMUP_STEPS="${COMPILER_WARMUP_STEPS:-0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-40}"

# ── Eval - Mode Dependent ───────────────────────────────────────────────────
if [[ "$MODE" == "validate" ]]; then
    export TURBO_QUANT_EXPORT=1
    export GPTQ_LITE_ENABLED="${GPTQ_LITE_ENABLED:-0}"
    export HESSIAN_TERNARY_GPTQ="${HESSIAN_TERNARY_GPTQ:-1}"
    export SELECTIVE_PRUNING="${SELECTIVE_PRUNING:-0}"
    export SLIDING_EVAL="${SLIDING_EVAL:-1}"
    export NGRAM_CACHE_ENABLED="${NGRAM_CACHE_ENABLED:-1}"
    export TEMP_SCALING="${TEMP_SCALING:-1}"
else
    export TURBO_QUANT_EXPORT=0
    export GPTQ_LITE_ENABLED=0
    export HESSIAN_TERNARY_GPTQ=0
    export SELECTIVE_PRUNING=0
    export SLIDING_EVAL=0
    export NGRAM_CACHE_ENABLED=0
    export TEMP_SCALING=0
fi

if [[ "$FAST_SMOKE" == "1" ]]; then
    export MAX_WALLCLOCK_SECONDS=45
    export VAL_LOSS_EVERY=0
    export TRAIN_LOG_EVERY=1
    export GPTQ_LITE_ENABLED=0
    export HESSIAN_TERNARY_GPTQ=0
    export SELECTIVE_PRUNING=0
    export SLIDING_EVAL=0
    export NGRAM_CACHE_ENABLED=0
    export TEMP_SCALING=0
fi

# ═════════════════════════════════════════════════════════════════════════════
# Per-config Architecture Overrides (MUST BE 128-ALIGNED)
# ═════════════════════════════════════════════════════════════════════════════
case "$CONFIG" in
    baseline) # 8L 512D 128C
        ;;
    skc_6L_512D_128C)
        export NUM_LAYERS=6
        export MODEL_DIM=512
        export SKC_CAPSULE_DIM=128
        ;;
    skc_7L_512D_128C)
        export NUM_LAYERS=7
        export MODEL_DIM=512
        export SKC_CAPSULE_DIM=128
        ;;
    skc_8L_512D_128C)
        export NUM_LAYERS=8
        export MODEL_DIM=512
        export SKC_CAPSULE_DIM=128
        ;;
    skc_9L_512D_128C)
        export NUM_LAYERS=9
        export MODEL_DIM=512
        export SKC_CAPSULE_DIM=128
        ;;
    skc_10L_512D_128C)
        export NUM_LAYERS=10
        export MODEL_DIM=512
        export SKC_CAPSULE_DIM=128
        ;;
    skc_10L_640D_128C)
        export NUM_LAYERS=10
        export MODEL_DIM=640
        export SKC_CAPSULE_DIM=128
        ;;
    skc_10L_640D_256C)
        export NUM_LAYERS=10
        export MODEL_DIM=640
        export SKC_CAPSULE_DIM=256
        ;;
    skc_12L_512D_128C)
        export NUM_LAYERS=12
        export MODEL_DIM=512
        export SKC_CAPSULE_DIM=128
        ;;
    skc_12L_640D_128C)
        export NUM_LAYERS=12
        export MODEL_DIM=640
        export SKC_CAPSULE_DIM=128
        ;;
    *) echo "Unknown config '$CONFIG'"; exit 1 ;;
esac

mkdir -p logs
echo "Running ${CONFIG} (L=${NUM_LAYERS} D=${MODEL_DIM} C=${SKC_CAPSULE_DIM} MODE=${MODE} SMOKE=${FAST_SMOKE})"
RUN_NAME="ablation_${CONFIG}${RUN_TAG:+_${RUN_TAG}}"
LOG="logs/${RUN_NAME}.log"
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 "${TRAINER_PATH}" 2>&1 | tee "$LOG"

cp final_model.ternary.ptz "logs/${RUN_NAME}_model.ternary.ptz" 2>/dev/null || true
cp submission.json "logs/${RUN_NAME}_submission.json" 2>/dev/null || true

echo "=== DONE ${CONFIG} ==="
echo "Log      : ${LOG}"
echo "Artifact : logs/${RUN_NAME}_model.ternary.ptz"
echo "Submit   : logs/${RUN_NAME}_submission.json"
