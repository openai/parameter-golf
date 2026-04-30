#!/bin/bash
# RUN_NO_TTT: Our best config — matches run3 (1.1496) exactly
#
# Our unique stack: VR + GA + Star-ReLU + per-layer lr + GradQuant + TrigramHash
# Proven config, no experiments. Revert after run6 showed sigmoid gates + decoder 2x LR + bigram 8192 hurt.

set -e
cd /workspace/parameter-golf

# Architecture (matches run3 exactly)
export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=4
export VE_ENABLED=1
export WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=524288
export GRAD_CLIP_NORM=0.3

# Our innovations
export VALUE_RESIDUAL=1
export GATED_ATTENTION=1
export STAR_RELU=1
export PERLAYER_TRAIN_LR=1
export PROJ_LR_MULT=1.5
export FC_LR_MULT=0.7
export TRIGRAM_HASH=1

# Run3 config — no sigmoid gates, no decoder 2x LR, bigram 4096
export SIGMOID_SKIP_GATES=0
export DECODER_LR_MULT=1.0
export BIGRAM_HASH_BUCKETS=4096

# EMA + SWA, no QAT
export EMA_ENABLED=1
export SWA=1
export QAT=0

# No TTT
export TTT_ENABLED=0
export TTT_CAUSAL=0

export SEED=${1:-1337}
export RUN_TAG="no_ttt_$(date +%Y%m%d_%H%M%S)"

# Clean env
unset QUANT_BITS RUN_ID TIER2_MODE MLP_HIDDEN \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 PRUNE_PCT \
  REPTILE_TTT TTT_TWO_PHASE TTT_EPOCHS TTT_MAX_STEPS

echo "=== NO TTT (RUN3 CONFIG) ==="
echo "SEED=$SEED VR=1 GA=1 STAR_RELU=1 PERLAYER_LR=1 TRIGRAM=1 bigram=4096 clip=0.3"
echo "============================="

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
