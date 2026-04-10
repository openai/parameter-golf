#!/bin/bash
# ============================================================================
# RUNPOD 8×H100 SXM — COMPETITION SUBMISSION (10 minutes / 16MB)
# Architecture: Spectral Koopman Capsule (SKC) + MoE
# ============================================================================
#
# Key design decisions:
#   ARCHITECTURE=skc 8L dim=1536: MoE gives effective parameter density vs depth.
#   EMA_ENABLED=1: only implemented weight-averaging mechanism in the trainer.
#   COMPILER_WARMUP_STEPS=20: pre-budget graph capture (outside 599s window).
#   WARMUP_STEPS=20: in-budget linear LR ramp (0 → base over first 20 steps).
#   CURRICULUM_ENABLED=1: context warmup (seq=64 -> 2048) allows for better
#     representation maturity before tackling long-range dependencies.
#     Jump to full 2048 sequence happens at 24% of wall-clock.
#   TTT_ENABLED=0, VAL_LOSS_EVERY=0: every ms of 599s budget is training compute.
#   TURBO_QUANT_TRAIN=1 + TURBO_QUANT_EXPORT=1: Hadamard rotation must match.
#
# Artifacts written by trainer:
#   final_model.ternary.ptz  — submission artifact
#   submission.json          — metadata / BPB summary
#
# Deploy (RunPod):
#   runpodctl create pod \
#     --gpuType "NVIDIA H100 80GB HBM3" --gpuCount 8 \
#     --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
#     --volumeSize 50
#
# Setup on pod:
#   pip install sentencepiece
#   pip install flash-attn --no-build-isolation
#   # Copy data to /workspace/data/ and code to /workspace/
#
# Run:
#   bash run_runpod_8xh100.sh
# ============================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
[[ -f "${PROJECT_ROOT}/train_gpt.py" ]] || { echo "ERROR: ${PROJECT_ROOT}/train_gpt.py not found" >&2; exit 1; }

# ── Data ─────────────────────────────────────────────────────────────────────
export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: TOKENIZER_PATH not found: ${TOKENIZER_PATH}" >&2; exit 1; }

# ── Architecture ─────────────────────────────────────────────────────────────
export ARCHITECTURE=skc
export NUM_LAYERS=8
export MODEL_DIM=1536
export NUM_HEADS=24
export NUM_KV_HEADS=6
export MLP_MULT=4
export EMBED_DIM=256
export PARTIAL_ROPE_DIMS=32

# MoE
export MOE_ENABLED=1
export MOE_NUM_EXPERTS=4
export MOE_TOP_K=1
export MOE_START_FRACTION=0.30
export MOE_ROUTER_AUX_LOSS_COEF=0.01
export MOE_LAYER_FRAC=0.67

# SKC
export SKC_BLOCK_SIZE=64
export SKC_NUM_CAPSULES=24
export SKC_CAPSULE_DIM=64
export SKC_CONV_KERNEL=4

# ── Training budget ───────────────────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-599}
export ITERATIONS=500000
export WARMUP_STEPS=20            # In-budget linear LR ramp (0 → base over first 20 steps)
export COMPILER_WARMUP_STEPS=20   # Pre-budget graph capture — state reset after, outside 599s
export SEED=${SEED:-42}

# ── Batch sizing ──────────────────────────────────────────────────────────────
export TRAIN_BATCH_TOKENS=262144
export TRAIN_SEQ_LEN=2048
export TRAINING_DEPTH_RECURRENCE=0

# ── Curriculum ────────────────────────────────────────────────────────────────
# Now enabled: though H100 handles full seq=2048, context warmup allows for better
# representation maturity before tackling long-range dependencies.
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

# ── Optimizer ────────────────────────────────────────────────────────────────
export MATRIX_OPTIMIZER=muon
export MATRIX_LR=0.02
export SCALAR_LR=0.015
export TIED_EMBED_LR=0.025
export HEAD_LR=0.015
export MUON_WD=0.090
export ADAM_WD=0.090
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=0
export MUON_BACKEND_STEPS=5
export GRAD_CLIP_NORM=0.3
export WARMDOWN_FRACTION=0.20

# ── Weight averaging ─────────────────────────────────────────────────────────
# EMA is the only implemented averaging mechanism in the trainer.
export EMA_ENABLED=1
export EMA_DECAY=0.997
export EMA_START_FRACTION=0.20    # Start EMA slightly later to avoid step-1 noise

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

# ── Disabled features ─────────────────────────────────────────────────────────
export CAPSULE_ENABLED=0
export KOOPMAN_ENABLED=0
export KOOPMAN_SPECULATOR_ENABLED=0
export FEEDBACK_ENABLED=0
export VRL_ENABLED=0
export TTT_ENABLED=0
export SHARED_BLOCKS=0
export WEIGHT_SHARING=0
export INSIDE_OUT_TRAINING=0
export DEQ_FEEDBACK=0
export XSA_START_LAYER=999
export STOCHASTIC_DEPTH_PROB=0
export SELF_DISTILL_KL_WEIGHT=0

# ── Eval stack ────────────────────────────────────────────────────────────────
export VAL_LOSS_EVERY=0           # No mid-training val — every ms is training
export TRAIN_LOG_EVERY=50
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE=64
export SLIDING_BATCH_SIZE=256
export TEMP_SCALING=1

# ── Ternary quantization ──────────────────────────────────────────────────────
export BITNET_GROUP_SIZE=128
export TURBO_QUANT_TRAIN=1        # Must match EXPORT — Hadamard rotation at both
export TURBO_QUANT_EXPORT=1
export TURBO_QUANT_KV=1
export EXPORT_ALIGNED_TRAIN=0
export EXPORT_ALIGNED_TRAIN_START_FRACTION=0.0
export TERNARY_THRESHOLD_SEARCH=0
export TERNARY_THRESHOLD_LOW=0.02
export TERNARY_THRESHOLD_HIGH=0.15
export TERNARY_THRESHOLD_STEPS=4
export TERNARY_SCALE_SEARCH=0
export TERNARY_SCALE_MULT_LOW=0.9
export TERNARY_SCALE_MULT_HIGH=1.1
export TERNARY_SCALE_MULT_STEPS=3
export TERNARY_CALIB_TOP_N=5
export EXPORT_PROXY_EVAL=1
export EXPORT_PROXY_EVERY=1200
export EXPORT_PROXY_NUM_SEQS=4
export LZMA_PRESET=3

# ── torch.compile ─────────────────────────────────────────────────────────────
export COMPILE_MODE=default

# ── NCCL (single-node 8×H100 NVLink) ─────────────────────────────────────────
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

# ── Run ───────────────────────────────────────────────────────────────────────
export RUN_ID="skc_h100_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
rm -f final_model.ternary.ptz submission.json
LOG="${DIR}/logs/${RUN_ID}.log"

echo "=========================================================================="
echo "  SKC Competition Run — 8×H100 SXM"
echo "  RUN ID : ${RUN_ID}"
echo "  MODEL  : SKC  L=${NUM_LAYERS}  D=${MODEL_DIM}  H=${NUM_HEADS}  MoE=${MOE_NUM_EXPERTS}x(top-${MOE_TOP_K})"
echo "  BUDGET : ${MAX_WALLCLOCK_SECONDS}s  (compiler_warmup=${COMPILER_WARMUP_STEPS} pre-budget, lr_warmup=${WARMUP_STEPS} in-budget)"
echo "  BATCH  : ${TRAIN_BATCH_TOKENS} tokens/step ($((TRAIN_BATCH_TOKENS/2048)) seqs/step, $((TRAIN_BATCH_TOKENS/2048/8)) seqs/GPU)"
echo "  LR     : matrix=${MATRIX_LR}  scalar=${SCALAR_LR}  warmdown_frac=${WARMDOWN_FRACTION}"
echo "  QUANT  : turbo_train=${TURBO_QUANT_TRAIN} turbo_export=${TURBO_QUANT_EXPORT} aligned=${EXPORT_ALIGNED_TRAIN}@${EXPORT_ALIGNED_TRAIN_START_FRACTION}"
echo "  CURR   : 64 -> 128 -> 256 -> 512 -> 1024 @ 24% / 76%"
echo "  AVGING : EMA decay=${EMA_DECAY} start=${EMA_START_FRACTION}"
echo "=========================================================================="

OMP_NUM_THREADS=1 \
TORCH_NCCL_TIMEOUT_SEC=7200 \
torchrun --standalone --nproc_per_node=8 "${PROJECT_ROOT}/"${TRAINER_PATH}"" 2>&1 | tee "$LOG"

# Trainer writes: final_model.ternary.ptz, submission.json
cp final_model.ternary.ptz "logs/${RUN_ID}_model.ternary.ptz" 2>/dev/null || true
cp submission.json         "logs/${RUN_ID}_submission.json"   2>/dev/null || true
cp final_model.ternary.ptz "logs/skc_h100_model.ternary.ptz" 2>/dev/null || true

echo "=== DONE ==="
echo "Log      : $LOG"
echo "Artifact : logs/${RUN_ID}_model.ternary.ptz"
