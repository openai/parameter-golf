#!/bin/bash
# RUN_NO_TTT_BEST: Run3 base + free lunches + QAT
#
# Run3 config (1.1496) + three free lunches from intel:
#   MATRIX_LR=0.03 (verified -0.005+ BPB)
#   LeakyReLU(0.5)^2 (zero params, -0.003 BPB proven)
#   QAT=1 (run5 proved negative quant gap)
#
# Real target: #445 at 1.1236 (not #505 which doesn't fit 16MB)
# Projected: ~1.140-1.145

set -e
cd /workspace/parameter-golf

# Architecture (run3 base)
export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=4
export VE_ENABLED=1
export WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=524288
export GRAD_CLIP_NORM=0.3
export BIGRAM_HASH_BUCKETS=4096

# Free lunch: MATRIX_LR=0.03 (PR #530, verified -0.005+ BPB)
export MATRIX_LR=0.03

# Our innovations
export VALUE_RESIDUAL=1
export GATED_ATTENTION=1
export PERLAYER_TRAIN_LR=1
export PROJ_LR_MULT=1.5
export FC_LR_MULT=0.7
export TRIGRAM_HASH=1

# Free lunch: LeakyReLU(0.5)^2 instead of Star-ReLU (zero params, -0.003 BPB)
export STAR_RELU=0
export LEAKY_RELU=1

# Not from run6 (those hurt without QAT protection)
export SIGMOID_SKIP_GATES=0
export DECODER_LR_MULT=1.0

# QAT=1 (run5 proved this fixes quant gap)
export QAT=1

# EMA + SWA
export EMA_ENABLED=1
export SWA=1

# No TTT
export TTT_ENABLED=0
export TTT_CAUSAL=0

export SEED=${1:-1337}
export RUN_TAG="best_$(date +%Y%m%d_%H%M%S)"

# Clean env
unset QUANT_BITS RUN_ID TIER2_MODE MLP_HIDDEN \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 PRUNE_PCT \
  REPTILE_TTT TTT_TWO_PHASE TTT_EPOCHS TTT_MAX_STEPS

echo "=== BEST CONFIG: Run3 + free lunches + QAT ==="
echo "SEED=$SEED MATRIX_LR=0.03 LEAKY_RELU=1 QAT=1 VR=1 GA=1 PERLAYER_LR=1 TRIGRAM=1"
echo "================================================"

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
