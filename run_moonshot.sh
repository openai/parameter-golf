#!/bin/bash
# RUN_MOONSHOT: Everything stacked — targeting 1.11x
# Branch: next-gen (has GPTQ-lite, Reptile TTT, Shared VE)
# Changes from oneshot:
#   GPTQ_LITE=1: optimal per-row clip search at export (default ON)
#   REPTILE_TTT=1: Reptile meta-learning before TTT (~60s, makes TTT 10x better)
#   VE_ENABLED=1: Shared Value Embedding on layers 9,10 (~163K params)
# Expected: ~1.115-1.125 (speculative — untested techniques)
# Risk: Higher — three untested features. If score regresses, isolate with:
#   run_moonshot_gptq.sh (GPTQ only)
#   run_moonshot_reptile.sh (Reptile only)
#   run_moonshot_ve.sh (VE only)

set -e
cd /workspace/parameter-golf
git fetch origin && git checkout next-gen && git reset --hard origin/next-gen

export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=32 DOC_ISOLATED_EVAL=0 SEED=1337
export QAT=0 TTT_MAX_STEPS=500 TTT_FREEZE_BLOCKS=1
export TRAIN_BATCH_TOKENS=786432
export EMA_ENABLED=0 SWA=1
export WARMDOWN_ITERS=20000
export PRUNE_PCT=3.0
export GPTQ_LITE=1
export REPTILE_TTT=1
export VE_ENABLED=1

unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE MLP_QUANT_BITS \
  USE_FA3 LATE_K_FP16

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
