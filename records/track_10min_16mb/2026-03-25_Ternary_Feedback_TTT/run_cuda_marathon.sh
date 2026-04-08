#!/bin/bash
# 6-Hour CUDA Marathon Runway - Ternary Koopman-Attention Hybrid (TKA-H)
# Target: 8xH100 Extrapolated Performance on Consumer GPU (1650 Ti / 3060)

export RUN_ID="cuda_marathon_$(date +%Y%m%d_%H%M%S)"
export DATA_PATH="./data/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model"

# --- Architecture (E_Shatter_Expectations_v5) ---
export ARCHITECTURE="hybrid"
export MODEL_DIM=128
export NUM_LAYERS=8
export SHARED_BLOCKS=2
export MOE_ENABLED=1
export MOE_NUM_EXPERTS=3
export MOE_TOP_K=1

# --- Memory & Duration (OOM Protection) ---
export MAX_WALLCLOCK_SECONDS=21600 # 6 Hours
export TRAIN_BATCH_TOKENS=32768
export VAL_BATCH_SIZE=131072
export ITERATIONS=50000
export COMPILE_MODE="default" # More stable for long runs than max-autotune

# --- Convergence & Stability (MLX Port) ---
export TERNARY_NOISE_SCALE=0.05
export STOCHASTIC_DEPTH_PROB=0.1
export SELF_DISTILL_KL_WEIGHT=0.1
export EMA_ENABLED=1
export EMA_DECAY=0.997
export EMA_EVAL_APPLY=1
export FEEDBACK_ENABLED=1
export FEEDBACK_PASSES=1

# --- Curriculum Sequence ---
export CURRICULUM_ENABLED=1
export TRAIN_SEQ_LEN=1024
export CURRICULUM_PHASE1_SEQ=256
export CURRICULUM_PHASE2_SEQ=512
export CURRICULUM_PHASE1_FRAC=0.2
export CURRICULUM_PHASE2_FRAC=0.5

# --- Optimization (NeoMuon) ---
export MATRIX_OPTIMIZER="muon"
export MATRIX_LR=0.04
export ADAM_LR=0.04
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_MOMENTUM_WARMUP_START=0.85

# --- Launch ---
echo "🚀 Launching 6-Hour CUDA Marathon: $RUN_ID"
echo "📍 Model: TKA-H | Dim: $MODEL_DIM | Layers: $NUM_LAYERS | Shared: $SHARED_BLOCKS"
echo "🔒 Limits: 6h / 32k tokens per step / OOM Protected"

python train_gpt.py 2>&1 | tee "logs/cuda/${RUN_ID}_marathon.log"
