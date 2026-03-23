#!/bin/bash
# RUN_NO_TTT_1GPU: Same stack as run_no_ttt.sh but for 1xH100
# Slower (1 GPU), but cheaper for testing. Not for competition submission.

set -e
cd /workspace/parameter-golf
git fetch origin && git checkout perlayer-lr-stack && git reset --hard origin/perlayer-lr-stack

export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=4
export VE_ENABLED=1
export WARMDOWN_ITERS=3500
export VALUE_RESIDUAL=1
export GATED_ATTENTION=1
export PERLAYER_TRAIN_LR=1
export PROJ_LR_MULT=1.5
export FC_LR_MULT=0.7
export STAR_RELU=1
export TRIGRAM_HASH=1

# EMA + SWA, no QAT, no TTT
export EMA_ENABLED=1
export SWA=1
export QAT=0
export TTT_ENABLED=0
export TTT_CAUSAL=0

# 1GPU: more wallclock needed, smaller batch
export MAX_WALLCLOCK_SECONDS=3600
export TRAIN_BATCH_TOKENS=524288

export SEED=${1:-1337}

# Clean env
unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 PRUNE_PCT \
  REPTILE_TTT TTT_TWO_PHASE TTT_EPOCHS TTT_MAX_STEPS

echo "=== NO TTT 1xGPU ==="
echo "SEED=$SEED stride=64 EMA=1 SWA=1 QAT=0 VR=1 GA=1 PERLAYER_LR=1 STAR_RELU=1 TRIGRAM=1"
echo "====================="

python3 records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
