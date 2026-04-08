#!/bin/bash
# ───────────────────────────────────────────────────────────────────────
# Overnight Laptop Training Configuration for MLX Version
# ───────────────────────────────────────────────────────────────────────
#
# This configuration is optimized for:
# - Apple Silicon (M1/M2/M3 MacBook)
# - 8-16GB unified memory
# - 8-12 hour overnight training windows
# - Maximize convergence within power budget
#
# Key improvements added to train_gpt_mlx.py:
#   - SmearGate: temporal position mixing after embeddings (+0.005 BPB)
#   - LAWA: Latest-A-Wins weight averaging (better generalization)
#   - SWA: Stochastic Weight Averaging during warmdown (smoother loss)
#
# ───────────────────────────────────────────────────────────────────────

# Core timing — adjust for your machine and overnight window
export MAX_WALLCLOCK_SECONDS=43200  # 12 hours: suitable for overnight run

# Training parameters — reduced for memory efficiency on laptop
export ITERATIONS=15000            # ~12h on M3 = ~1.25 steps/sec
export TRAIN_BATCH_TOKENS=393216   # Halved from default 786432 for memory
export GRAD_ACCUM_STEPS=2           # Reduced from 4 to fit in memory
export TRAIN_SEQ_LEN=1024           # Reduced from 2048 to save memory

# Model architecture — optimized for laptop inference speed later
export NUM_LAYERS=10                # Reduced from 12 for faster training/inference
export MODEL_DIM=512                # Reduced from 768 for memory efficiency
export NUM_HEADS=8                  # Match num_kv_heads
export NUM_KV_HEADS=4
export MLP_MULT=4
export EMBED_DIM=128                # Reduced from 254 for memory

# Learning rates — tuned for smaller batch size
export MATRIX_LR=0.040              # Slightly higher for smaller batch
export SCALAR_LR=0.030
export TIED_EMBED_LR=0.040

# Convergence optimizations
export WARMUP_STEPS=10              # Reduced for faster convergence
export WARMDOWN_FRACTION=0.4        # More aggressive warmdown
export CURRICULUM_ENABLED=1
export CURRICULUM_PHASE1_FRAC=0.5
export CURRICULUM_PHASE2_FRAC=0.8
export CURRICULUM_PHASE1_SEQ=256
export CURRICULUM_PHASE2_SEQ=512

# NEW: Weight averaging improvements (added to MLX version)
export SMEARGATE_ENABLED=1          # Temporal position mixing
export LAWA_ENABLED=1               # Latest-A-Wins averaging
export LAWA_K=10                    # Keep 10 snapshots
export LAWA_FREQ=200                # Snapshot every 200 steps
export SWA_ENABLED=1                # Stochastic Weight Averaging
export SWA_EVERY=50                 # Accumulate every 50 steps
export SWA_START_SCALE=0.2          # During warmdown (scale < 0.2)

# EMA and advanced convergence
export EMA_ENABLED=1
export EMA_DECAY=0.9975             # More aggressive decay for overnight run
export EMA_START_FRACTION=0.5
export EMA_EVAL_APPLY=1

# Capsule and feedback (TKC architecture is non-negotiable)
export FEEDBACK_ENABLED=1
export FEEDBACK_PASSES=1
export FEEDBACK_EVERY=1
export CAPSULE_ENABLED=1
export CAPSULE_NUM=16
export CAPSULE_DIM=64
export KOOPMAN_ENABLED=1
export KOOPMAN_RANK=4

# Advanced features
export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_BUCKETS=3072
export BIGRAM_HASH_DIM=112
export XSA_START_LAYER=0            # XSA on all layers
export VRL_ENABLED=1
export VRL_START_LAYER=8
export PARTIAL_ROPE_DIMS=16
export ACTIVATION=lrelu2

# Ternary quantization and export
export GPTQ_LITE_ENABLED=1
export GPTQ_LITE_PERCENTILES=5
export TURBO_QUANT_EXPORT=1
export TURBO_QUANT_TRAIN=0

# Evaluation and validation
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=500
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE=128
export SLIDING_BATCH_SIZE=16
export TEMP_SCALING=1
export TTT_ENABLED=1

# MLX-specific optimizations
export MLX_MAX_MICROBATCH_TOKENS=4096
export MLX_EAGER_EVAL=0             # Use lazy evaluation for speed

# Paths
export DATA_PATH="${DATA_PATH:-.../data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-.../data/tokenizers/fineweb_8192_bpe.model}"
export OUT_DIR="logs/overnight_laptop_$(date +%Y%m%d_%H%M%S)"

# Run ID for logging
export RUN_ID="overnight_laptop_mlx_$(date +%Y%m%d_%H%M%S)"

# ───────────────────────────────────────────────────────────────────────
# USAGE:
#
#   source OVERNIGHT_LAPTOP_CONFIG.sh
#   python train_gpt_mlx.py
#
# Or inline:
#
#   source OVERNIGHT_LAPTOP_CONFIG.sh && python train_gpt_mlx.py
#
# ───────────────────────────────────────────────────────────────────────
