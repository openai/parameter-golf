#!/bin/bash
# RUN_CAUSAL_TTT: Score-first causal TTT (legal per FAQ)
#
# Each val chunk is scored under inference_mode BEFORE training on it.
# Compliant with: "you are only allowed to test-time train on validation
# set tokens you've already evaluated your model on"
#
# Uses cosine per-layer schedule within each chunk's training phase.
# Expected TTT gain: ~0.002-0.005 BPB (vs 0.06 for train-all-then-score)

set -e
cd /workspace/parameter-golf
git fetch origin && git checkout causal-ttt && git reset --hard origin/causal-ttt

export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export QAT=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=0

# Causal TTT config
export TTT_CAUSAL=1
export TTT_CHUNK_TOKENS=32768
export TTT_CHUNK_EPOCHS=3
export TTT_OPTIMIZER=adamw
export TTT_LR=0.0005
export TTT_COSINE=1
export TTT_PERLAYER=1
export TTT_FREEZE_BLOCKS=0
export TTT_BATCH_SEQS=64

export SEED=${1:-1337}

unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  WARMDOWN_ITERS BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 TRAIN_BATCH_TOKENS EMA_ENABLED SWA PRUNE_PCT \
  REPTILE_TTT VE_ENABLED TTT_TWO_PHASE TTT_EPOCHS TTT_MAX_STEPS

echo "=== CAUSAL TTT (LEGAL) ==="
echo "SEED=$SEED chunk=${TTT_CHUNK_TOKENS} epochs/chunk=${TTT_CHUNK_EPOCHS} cosine perlayer"
echo "=========================="

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
