#!/bin/bash
# ============================================================================
# SKC COMPETITION PROXY — 2-GPU (SP8192, virtual depth, score-first TTT)
# Branch: skc_competition_sp8192
# ============================================================================
#
# Mirrors run_skc_competition_8xh100.sh but scaled down for 2-GPU cheapness.
# Differences from the H100 script:
#   - MODEL_DIM=640 (5×128, tensor-core aligned for A40 class GPUs)
#   - NUM_HEADS=8, NUM_KV_HEADS=2
#   - nproc_per_node=2
#   - TTT_BATCH_SEQS=16 (half, due to smaller batch)
#   - VAL_LOSS_EVERY=0, no mid-training validation
#
# Purpose: proxy run for ablation + hyperparameter sweeps before committing
# to an expensive H100 run. Results should be directionally correct on:
#   - SP8192 vs SP1024 BPB delta
#   - Recurrence contribution
#   - TTT BPB improvement
#   - Export size with competition_gptq vs ternary_lzma
#
# Run locally (2 GPUs):
#   bash run_skc_competition_2gpu_proxy.sh
# Run on RunPod 2-GPU pod:
#   bash orchestrate_small_skc_multigpu_runpod.sh
# ============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${DIR}/../../.." && pwd)}"
TRAINER_PATH="${TRAINER_PATH:-train_gpt.py}"
[[ -f "${PROJECT_ROOT}/${TRAINER_PATH}" ]] || { echo "ERROR: ${PROJECT_ROOT}/${TRAINER_PATH} not found" >&2; exit 1; }

# ── Tokenizer regime: SP8192 ──────────────────────────────────────────────────
export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_8192_bpe.model}"
export VOCAB_SIZE=8192
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: TOKENIZER_PATH not found: ${TOKENIZER_PATH}" >&2; exit 1; }

# ── Architecture: SKC competition (2-GPU scale) ───────────────────────────────
export ARCHITECTURE=skc_competition
export NUM_LAYERS=8
export MODEL_DIM=640           # 5×128, tensor-core aligned
export NUM_HEADS=8
export NUM_KV_HEADS=2
export MLP_MULT=4  # entropy reduction: MLP_MULT=4 with WD=0.090 targets high sparsity for compression
export EMBED_DIM=128
export PARTIAL_ROPE_DIMS=16

export SKC_PARALLEL_RESIDUAL=1
export SKC_BLOCK_SIZE=64
export SKC_NUM_CAPSULES=16
export SKC_CAPSULE_DIM=64
export SKC_CONV_KERNEL=4

export MOE_ENABLED=1
export MOE_NUM_EXPERTS=4
export MOE_TOP_K=1
export MOE_START_FRACTION=0.30
export MOE_ROUTER_AUX_LOSS_COEF=0.01
export MOE_LAYER_FRAC=0.67

export CAPSULE_ENABLED=0
export KOOPMAN_ENABLED=0
export KOOPMAN_SPECULATOR_ENABLED=0
export FEEDBACK_ENABLED=0
export VRL_ENABLED=0
export SHARED_BLOCKS=0
export INSIDE_OUT_TRAINING=0
export DEQ_FEEDBACK=0
export XSA_START_LAYER=999

# ── Virtual depth ─────────────────────────────────────────────────────────────
export TRAINING_DEPTH_RECURRENCE=0
export RECURRENCE_DEPTH=2
export RECURRENCE_START_FRACTION=0.35

# ── Training budget ───────────────────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-599}
export ITERATIONS=500000
export WARMUP_STEPS=20
export COMPILER_WARMUP_STEPS=20
export SEED=${SEED:-42}

# ── Batch sizing ──────────────────────────────────────────────────────────────
export TRAIN_BATCH_TOKENS=131072   # half of H100 batch (2 GPUs)
export TRAIN_SEQ_LEN=2048

# ── Curriculum ────────────────────────────────────────────────────────────────
export CURRICULUM_ENABLED=1
export CURRICULUM_PHASE1_FRAC=0.04
export CURRICULUM_PHASE2_FRAC=0.08
export CURRICULUM_PHASE3_FRAC=0.13
export CURRICULUM_PHASE4_FRAC=0.18
export CURRICULUM_PHASE5_FRAC=0.24
export CURRICULUM_PHASE1_SEQ=64
export CURRICULUM_PHASE2_SEQ=128
export CURRICULUM_PHASE3_SEQ=256
export CURRICULUM_PHASE4_SEQ=512
export CURRICULUM_PHASE5_SEQ=1024

# ── Optimizer: frontier operating point ──────────────────────────────────────
export MATRIX_OPTIMIZER=muon
export MATRIX_LR=0.025
export SCALAR_LR=0.018
export TIED_EMBED_LR=0.030
export HEAD_LR=0.018
export MUON_WD=0.090
export ADAM_WD=0.090
export QK_GAIN_INIT=3.0
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=0
export MUON_BACKEND_STEPS=5
export GRAD_CLIP_NORM=0.3
export WARMDOWN_FRACTION=0.15

# ── Weight averaging ─────────────────────────────────────────────────────────
export EMA_ENABLED=1
export EMA_DECAY=0.997
export EMA_START_FRACTION=0.20

# ── Engram hash ───────────────────────────────────────────────────────────────
export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_BUCKETS=3072   # EngramLite
export BIGRAM_HASH_DIM=112
export ENGRAM_NUM_HEADS=2         # 2×2 = 4 total → unrolled fast-path in forward()
export ENGRAM_NUM_ORDERS=2
export ENGRAM_INJECT_LAYER=1

# ── N-gram cache ──────────────────────────────────────────────────────────────
export NGRAM_CACHE_ENABLED=1
export NGRAM_MAX_ORDER=5
export NGRAM_ALPHA_BASE=0.05
export NGRAM_ALPHA_SCALE=0.55
export NGRAM_ENTROPY_CENTER=4.0

# ── Eval stack ────────────────────────────────────────────────────────────────
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=50
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE=64
export SLIDING_BATCH_SIZE=128
export TEMP_SCALING=1

# ── Legal score-first TTT ─────────────────────────────────────────────────────
export TTT_ENABLED=1
export TTT_SCOPE=skc_safe
export TTT_LR=0.003
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=16      # scaled down for 2-GPU proxy
export TTT_GRAD_CLIP=1.0

# ── Competition export ────────────────────────────────────────────────────────
export EXPORT_MODE=competition_gptq
export GPTQ_LITE_ENABLED=1
export GPTQ_LITE_PERCENTILES=5
export BITNET_GROUP_SIZE=128
export TURBO_QUANT_TRAIN=1
export TURBO_QUANT_EXPORT=1
export TURBO_QUANT_KV=1
export EXPORT_ALIGNED_TRAIN=0
export TERNARY_THRESHOLD_SEARCH=0
export TERNARY_SCALE_SEARCH=0
export TERNARY_CALIB_TOP_N=5
export EXPORT_PROXY_EVAL=1
export EXPORT_PROXY_EVERY=1200
export EXPORT_PROXY_NUM_SEQS=4
export LZMA_PRESET=3

# ── torch.compile ─────────────────────────────────────────────────────────────
export COMPILE_MODE=default

NPROC=${NPROC:-2}
export RUN_ID="skc_competition_proxy_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
rm -f final_model.ternary.ptz final_model.competition.ptz submission.json
LOG="${DIR}/logs/${RUN_ID}.log"

echo "=========================================================================="
echo "  SKC COMPETITION PROXY — ${NPROC}×GPU"
echo "  RUN_ID    : ${RUN_ID}"
echo "  TOKENIZER : SP${VOCAB_SIZE}"
echo "  MODEL     : skc_competition  L=${NUM_LAYERS}  D=${MODEL_DIM}  H=${NUM_HEADS}"
echo "              parallel_residual=1  recurrence=depth${RECURRENCE_DEPTH}@${RECURRENCE_START_FRACTION}"
echo "  TTT       : scope=${TTT_SCOPE}  epochs=${TTT_EPOCHS}  lr=${TTT_LR}"
echo "  EXPORT    : ${EXPORT_MODE}"
echo "  BUDGET    : ${MAX_WALLCLOCK_SECONDS}s"
echo "=========================================================================="

OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=${NPROC} "${PROJECT_ROOT}/${TRAINER_PATH}" 2>&1 | tee "${LOG}"

cp final_model.ternary.ptz      "logs/${RUN_ID}_model.ternary.ptz"      2>/dev/null || true
cp final_model.competition.ptz  "logs/${RUN_ID}_model.competition.ptz"  2>/dev/null || true
cp submission.json               "logs/${RUN_ID}_submission.json"        2>/dev/null || true

echo "=== DONE ==="
echo "Log : ${LOG}"
