#!/bin/bash
# v13i: Strip both + wider model (dim=528) — trade size for width
#
# Base: v13g (strip both VE+bigram, 14.82 MiB)
# Change: MODEL_DIM=528 (was 512), TARGET_MB=15.25 (max safe)
# head_dim=66 (even, valid for RoPE), kv_dim=264
# Estimate: ~15.65 MiB unpruned → needs ~2.6% pruning
# Risk: similar pruning % to v13b, but with real capacity gain
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v13i_wider528"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# Architecture — strip both, widen to 528
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export MATRIX_CLIP_SIGMAS=12.85
export PARTIAL_KEY_OFFSET=1
export RECUR_LAYERS="4,5"
export RECUR_START_FRAC=0.5
export WARMDOWN_ITERS=4000
export VE_ENABLED=0
export BIGRAM_VOCAB_SIZE=0
export BIGRAM_DIM=112

# v13i change: wider model
export MODEL_DIM=528
export NUM_LAYERS=11

# Use maximum safe TARGET_MB
export TARGET_MB=15.25

# No TTT, no parallel
export TTT_ENABLED=0
export PARALLEL_START_LAYER=-1

echo "========================================"
echo "  v13i: Wider model (dim=528, strip both)"
echo "  MODEL_DIM=528, TARGET_MB=15.25"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_v13.py 2>&1 | tee "${LOGDIR}/v13i_wider528.log"
