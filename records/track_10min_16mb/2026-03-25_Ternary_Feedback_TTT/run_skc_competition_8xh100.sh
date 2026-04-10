#!/bin/bash
# ============================================================================
# SKC COMPETITION — 8×H100 SXM (SP8192, virtual depth, score-first TTT)
# Branch: skc_competition_sp8192
# ============================================================================
#
# What changed from run_runpod_8xh100.sh (research/sp1024 baseline):
#   SP8192  : VOCAB_SIZE=8192, fineweb10B_sp8192 data, fineweb_8192_bpe.model
#   Recurrence : RECURRENCE_DEPTH=2, RECURRENCE_START_FRACTION=0.35
#                (virtual depth activates at 35% of wall-clock)
#   TTT     : TTT_ENABLED=1, TTT_SCOPE=skc_safe (score-first, legal)
#             3 epochs/chunk, SGD+momentum, chunk=32K tokens
#   Export  : EXPORT_MODE=competition_gptq, GPTQ_LITE_ENABLED=1
#             brotli-first compression (falls back to LZMA if brotli absent)
#   Parallel: SKC_PARALLEL_RESIDUAL=1 (parallel SKC+MLP paths, merged residual)
#   HP      : frontier operating point (higher QK, WD, later warmdown)
#   MoE     : kept (re-qualified after SP8192 migration)
#
# Architecture: SKC 8L dim=1536, MoE 4 experts, ParallelSKCBlock, SP8192
#
# Prerequisites on the RunPod pod:
#   pip install sentencepiece
#   pip install flash-attn --no-build-isolation
#   pip install brotli  # optional, falls back to LZMA if absent
#   # Tokenizer: /workspace/data/tokenizers/fineweb_8192_bpe.model
#   # Data:      /workspace/data/datasets/fineweb10B_sp8192/
#
# Run:
#   bash run_skc_competition_8xh100.sh
# ============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${DIR}/../../.." && pwd)"
TRAINER_PATH="train_gpt.py"
[[ -f "${PROJECT_ROOT}/${TRAINER_PATH}" ]] || { echo "ERROR: ${PROJECT_ROOT}/${TRAINER_PATH} not found" >&2; exit 1; }

# ── Tokenizer regime: SP8192 ──────────────────────────────────────────────────
export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_8192_bpe.model}"
# VOCAB_SIZE is auto-detected from the tokenizer model; set here for clarity
export VOCAB_SIZE=8192
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2; echo "  Expected SP8192 data at ${DATA_PATH}" >&2; echo "  Run tokenization script or mount SP8192 dataset." >&2; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: TOKENIZER_PATH not found: ${TOKENIZER_PATH}" >&2; exit 1; }

# ── Architecture: SKC competition profile ────────────────────────────────────
export ARCHITECTURE=skc_competition   # maps to 'skc' with competition context
export NUM_LAYERS=8
export MODEL_DIM=1536
export NUM_HEADS=24
export NUM_KV_HEADS=6
export MLP_MULT=3
export EMBED_DIM=256
export PARTIAL_ROPE_DIMS=32

# Parallel residual SKC (two paths merge before residual add)
export SKC_PARALLEL_RESIDUAL=1
export SKC_BLOCK_SIZE=64
export SKC_NUM_CAPSULES=24
export SKC_CAPSULE_DIM=64
export SKC_CONV_KERNEL=4

# MoE — re-qualified: keeps positive contribution at SP8192
export MOE_ENABLED=1
export MOE_NUM_EXPERTS=4
export MOE_TOP_K=1
export MOE_START_FRACTION=0.30
export MOE_ROUTER_AUX_LOSS_COEF=0.01
export MOE_LAYER_FRAC=0.67

# Disabled research surface
export CAPSULE_ENABLED=0
export KOOPMAN_ENABLED=0
export KOOPMAN_SPECULATOR_ENABLED=0
export FEEDBACK_ENABLED=0
export VRL_ENABLED=0
export SHARED_BLOCKS=0
export WEIGHT_SHARING=0
export INSIDE_OUT_TRAINING=0
export DEQ_FEEDBACK=0
export XSA_START_LAYER=999
export STOCHASTIC_DEPTH_PROB=0
export SELF_DISTILL_KL_WEIGHT=0

# ── Virtual depth: recurrence ─────────────────────────────────────────────────
# Off during early training; activates at 35% of wall-clock for stability.
# TRAINING_DEPTH_RECURRENCE=0 keeps the initial compiled graph simple.
# Recurrence scheduling in train_gpt.py flips backbone.training_depth_recurrence
# to RECURRENCE_DEPTH at the configured fraction.
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
export TRAIN_BATCH_TOKENS=262144
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
# Moved from conservative research defaults toward the leaderboard-competitive
# regime: higher QK gain, higher WD, later warmdown start.
export MATRIX_OPTIMIZER=muon
export MATRIX_LR=0.025
export SCALAR_LR=0.018
export TIED_EMBED_LR=0.030
export HEAD_LR=0.018
export MUON_WD=0.08          # up from 0.04 (research default)
export ADAM_WD=0.08
export QK_GAIN_INIT=3.0      # up from 2.25 (research default)
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=0
export MUON_BACKEND_STEPS=5
export GRAD_CLIP_NORM=0.3
export WARMDOWN_FRACTION=0.15  # later warmdown than research (0.20→0.15 of budget)

# ── Weight averaging ─────────────────────────────────────────────────────────
export EMA_ENABLED=1
export EMA_DECAY=0.997
export EMA_START_FRACTION=0.20

# ── Engram hash ───────────────────────────────────────────────────────────────
export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_BUCKETS=8192
export BIGRAM_HASH_DIM=48
export ENGRAM_NUM_HEADS=4
export ENGRAM_NUM_ORDERS=2
export ENGRAM_INJECT_LAYER=1

# ── N-gram cache ──────────────────────────────────────────────────────────────
export NGRAM_CACHE_ENABLED=1
export NGRAM_MAX_ORDER=5
export NGRAM_ALPHA_BASE=0.05
export NGRAM_ALPHA_SCALE=0.55
export NGRAM_ENTROPY_CENTER=4.0

# ── Eval stack ────────────────────────────────────────────────────────────────
export VAL_LOSS_EVERY=0        # no mid-training val — every ms is training
export TRAIN_LOG_EVERY=50
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE=64
export SLIDING_BATCH_SIZE=256
export TEMP_SCALING=1

# ── Legal score-first TTT ─────────────────────────────────────────────────────
# Score-first semantics: evaluate under no-grad, then update, then next chunk.
# Scope=skc_safe: decay_rates + residual mix + SKC scales (no weight matrices).
# Timing enforced by eval budget — TTT cannot overflow the leaderboard window.
export TTT_ENABLED=1
export TTT_SCOPE=skc_safe
export TTT_LR=0.003
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

# ── Competition export: GPTQ-lite + brotli ────────────────────────────────────
# EXPORT_MODE=competition_gptq enables:
#   1. GPTQ-lite per-row clip search before ternary quantization
#   2. brotli compression (falls back to LZMA if brotli package absent)
#   3. Writes both final_model.ternary.ptz and final_model.competition.ptz
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

# ── NCCL ─────────────────────────────────────────────────────────────────────
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

# ── Run ───────────────────────────────────────────────────────────────────────
export RUN_ID="skc_competition_sp8192_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
rm -f final_model.ternary.ptz final_model.competition.ptz submission.json
LOG="${DIR}/logs/${RUN_ID}.log"

echo "=========================================================================="
echo "  SKC COMPETITION RUN — 8×H100 SXM"
echo "  RUN_ID    : ${RUN_ID}"
echo "  TOKENIZER : SP${VOCAB_SIZE}  (competition regime)"
echo "  MODEL     : skc_competition  L=${NUM_LAYERS}  D=${MODEL_DIM}  H=${NUM_HEADS}"
echo "              MoE=${MOE_NUM_EXPERTS}x(top-${MOE_TOP_K})  parallel_residual=1"
echo "  RECURRENCE: depth=${RECURRENCE_DEPTH} @ frac=${RECURRENCE_START_FRACTION}"
echo "  TTT       : enabled  scope=${TTT_SCOPE}  epochs=${TTT_EPOCHS}  lr=${TTT_LR}"
echo "  EXPORT    : ${EXPORT_MODE}  gptq_lite=${GPTQ_LITE_ENABLED}"
echo "  BUDGET    : ${MAX_WALLCLOCK_SECONDS}s  compiler_warmup=${COMPILER_WARMUP_STEPS}"
echo "  LR        : matrix=${MATRIX_LR}  scalar=${SCALAR_LR}  qk_gain=${QK_GAIN_INIT}"
echo "  WD/warmdn : muon_wd=${MUON_WD}  warmdown_frac=${WARMDOWN_FRACTION}"
echo "  DATA      : ${DATA_PATH}"
echo "  CURRICULUM: 64->128->256->512->1024 @ 24% / 76%"
echo "=========================================================================="

OMP_NUM_THREADS=1 \
TORCH_NCCL_TIMEOUT_SEC=7200 \
torchrun --standalone --nproc_per_node=8 "${PROJECT_ROOT}/${TRAINER_PATH}" 2>&1 | tee "${LOG}"

# Archive artifacts
cp final_model.ternary.ptz      "logs/${RUN_ID}_model.ternary.ptz"      2>/dev/null || true
cp final_model.competition.ptz  "logs/${RUN_ID}_model.competition.ptz"  2>/dev/null || true
cp submission.json               "logs/${RUN_ID}_submission.json"        2>/dev/null || true

echo "=== DONE ==="
echo "Log      : ${LOG}"
echo "Artifact : logs/${RUN_ID}_model.ternary.ptz"
