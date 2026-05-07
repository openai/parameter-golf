#!/bin/bash
# ============================================================================
# RUNPOD 8×H100 SXM — COMPETITION SUBMISSION (10 minutes / 16MB)
# ============================================================================
# Deploy: runpodctl create pod --gpuType "NVIDIA H100 80GB HBM3" --gpuCount 8
#   --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   --volumeSize 50
#
# Setup on pod:
#   pip install sentencepiece
#   # Upload data to /workspace/data/ or use setup.sh
#
# Then run:
#   bash run_runpod_8xh100.sh
# ============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

# ── Data paths (RunPod workspace) ─────────────────────────────────────────
export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_1024_bpe.model}"

# ── Architecture: TKA-H v5 (proven champion from MLX sweep) ──────────────
export ARCHITECTURE=hybrid
export NUM_LAYERS=8
export MODEL_DIM=768
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export EMBED_DIM=254
export SHARED_BLOCKS=2
export VOCAB_SIZE=1024

# ── Training budget ──────────────────────────────────────────────────────
export ITERATIONS=10000
export MAX_WALLCLOCK_SECONDS=599
export WARMUP_STEPS=5
export SEED=42

# ── Batch sizing (8×H100 = massive throughput) ───────────────────────────
# 786K global tokens → 98K per GPU. seq_len=2048 → 48 seqs/GPU.
# H100 80GB handles this easily in bf16.
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048

# ── Optimizer: Tuned from HP sweep ───────────────────────────────────────
# LR=0.035 was optimal in 1hr MLX sweep. For H100 with larger model (768 vs 128),
# scale down slightly per mu-parameterization: 0.035 * sqrt(128/768) ≈ 0.014.
# But competition is 10-min (not 1hr), so keep higher for faster convergence.
export MATRIX_OPTIMIZER=muon
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export ADAM_LR=0.025
export ADAM_WD=0.04
export MUON_WD=0.04
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_BACKEND_STEPS=5
export GRAD_CLIP_NORM=0.3
export WARMDOWN_FRACTION=0.5

# ── Curriculum (fast, tuned for 10-min window) ───────────────────────────
# Phase transitions by wallclock fraction.
# 10 min = 600s. Phase1 ends at 90s (seq=256), Phase2 at 240s (seq=512),
# rest at seq=2048 (full). H100 steps are fast so model adapts quickly.
export CURRICULUM_ENABLED=1
export CURRICULUM_PHASE1_SEQ=256
export CURRICULUM_PHASE2_SEQ=512
export CURRICULUM_PHASE1_FRAC=0.15
export CURRICULUM_PHASE2_FRAC=0.40

# ── Feedback (disabled during training, 2 passes at eval) ────────────────
export FEEDBACK_ENABLED=0
export FEEDBACK_DIM=32
export FEEDBACK_SKETCH_TOKENS=2
export FEEDBACK_PASSES=1
export EVAL_FEEDBACK_PASSES=2
export FEEDBACK_EVERY=2

# ── Capsules & Koopman ───────────────────────────────────────────────────
export CAPSULE_ENABLED=1
export CAPSULE_NUM=16
export CAPSULE_DIM=64
export CAPSULE_CARRY_ENABLED=1
export CAPSULE_CARRY_DECAY=0.8
export KOOPMAN_ENABLED=1
export KOOPMAN_RANK=2
export KOOPMAN_DIAG_INIT=0.9
export KOOPMAN_CONSISTENCY_WEIGHT=0.005
export KOOPMAN_SPECULATOR_ENABLED=1
export KOOPMAN_SPECULATOR_STEPS=3
export KOOPMAN_SPECULATOR_WEIGHT=0.01
export ADAPTIVE_HALT_ENABLED=1
export ADAPTIVE_HALT_THRESHOLD=0.05
export MAX_EVAL_PASSES=3

# ── Koopman SSM ──────────────────────────────────────────────────────────
export KOOPMAN_STATE_DIM=128
export KOOPMAN_MIXER_RANK=4
export KOOPMAN_CONV_KERNEL=4
export KOOPMAN_DECAY_WINDOW=32

# ── MoE (3 experts, top-1 → 3× FFN params, same FLOPs) ─────────────────
export MOE_ENABLED=1
export MOE_NUM_EXPERTS=3
export MOE_TOP_K=1
export MOE_ROUTER_AUX_LOSS_COEF=0.01

# ── Engram Hash ──────────────────────────────────────────────────────────
export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_BUCKETS=4096
export BIGRAM_HASH_DIM=64
export ENGRAM_NUM_HEADS=4
export ENGRAM_NUM_ORDERS=3
export ENGRAM_INJECT_LAYER=1

# ── Convergence features ─────────────────────────────────────────────────
export STOCHASTIC_DEPTH_PROB=0.1
export TERNARY_NOISE_SCALE=0.02
export SELF_DISTILL_KL_WEIGHT=0.1

# ── Attention & normalization ─────────────────────────────────────────────
export LOGIT_SOFTCAP=30
export SOFTCAP_TYPE=poly
export QK_GAIN_INIT=2.25
export ACTIVATION=lrelu2
export LEAKY_RELU_SLOPE=0.5
export ROPE_BASE=5000
export ROPE_TYPE=yarn
export YARN_MAX_LEN=4096
export TIE_EMBEDDINGS=1
export BITNET_GROUP_SIZE=128
export VRL_ENABLED=1
export VRL_START_LAYER=10
export LN_SCALE_DAMPING=1
export PARTIAL_ROPE_DIMS=16
export XSA_START_LAYER=8

# ── EMA ──────────────────────────────────────────────────────────────────
export EMA_ENABLED=1
export EMA_EVAL_APPLY=1
export EMA_DECAY=0.997
export EMA_START_FRACTION=0.40

# ── Eval stack ───────────────────────────────────────────────────────────
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE=64
export SLIDING_BATCH_SIZE=256
export TEMP_SCALING=1
export TURBO_QUANT_EXPORT=1
export TURBO_QUANT_TRAIN=0

# ── TTT & N-gram cache ──────────────────────────────────────────────────
export TTT_ENABLED=1
export TTT_SCOPE=feedback
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0
export NGRAM_CACHE_ENABLED=1
export NGRAM_MAX_ORDER=5
export NGRAM_ALPHA_BASE=0.05
export NGRAM_ALPHA_SCALE=0.55
export NGRAM_ENTROPY_CENTER=4.0

# ── Logging ──────────────────────────────────────────────────────────────
export VAL_LOSS_EVERY=100
export TRAIN_LOG_EVERY=10
export RUN_ID=h100_competition_$(date +%Y%m%d_%H%M%S)

# ── torch.compile (H100 supports it natively, huge speedup) ──────────────
export COMPILE_MODE=default

# ── Launch (8 GPU DDP via torchrun) ──────────────────────────────────────
LOG_FILE="logs/h100_${RUN_ID}.log"
mkdir -p logs

echo "=========================================================================="
echo "LAUNCHING: TKA-H v5 Competition Run on 8×H100 SXM"
echo "RUN ID: ${RUN_ID}"
echo "MODEL: hybrid L=${NUM_LAYERS} D=${MODEL_DIM} H=${NUM_HEADS} shared=${SHARED_BLOCKS}"
echo "BATCH: ${TRAIN_BATCH_TOKENS} tokens, seq=${TRAIN_SEQ_LEN}"
echo "BUDGET: ${MAX_WALLCLOCK_SECONDS}s wallclock"
echo "CURRICULUM: ${CURRICULUM_PHASE1_SEQ}→${CURRICULUM_PHASE2_SEQ}→${TRAIN_SEQ_LEN} @ ${CURRICULUM_PHASE1_FRAC}/${CURRICULUM_PHASE2_FRAC}"
echo "LR: matrix=${MATRIX_LR} scalar=${SCALAR_LR} embed=${TIED_EMBED_LR}"
echo "=========================================================================="

OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG_FILE"

echo "--- RUN COMPLETE ---"
echo "Log: $LOG_FILE"
echo "Artifact: logs/${RUN_ID}_model.ternary.ptz"
