#!/bin/bash
# v13g: Strip BOTH VE and bigram — maximum size headroom
#
# Changes from v13a: BIGRAM_VOCAB_SIZE=0 (v13a already has VE_ENABLED=0)
# Saves ~1.4 MB total (1.0 VE + 0.4 bigram) → ~14.1 MB unpruned
#
# Purpose: confirm no compound BPB cost from stripping both features.
# If free, we have ~1.9 MB headroom for capacity experiments.
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v13g_strip_both"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# Architecture — strip BOTH VE and bigram
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export MATRIX_CLIP_SIGMAS=12.85
export PARTIAL_KEY_OFFSET=1
export RECUR_LAYERS="4,5"
export RECUR_START_FRAC=0.5
export WARMDOWN_ITERS=4000

# v13g: both stripped
export VE_ENABLED=0
export BIGRAM_VOCAB_SIZE=0
export BIGRAM_DIM=112

export TARGET_MB=15.2

# No TTT, no parallel
export TTT_ENABLED=0
export PARALLEL_START_LAYER=-1

echo "========================================"
echo "  v13g: Strip BOTH VE + Bigram"
echo "  VE_ENABLED=0, BIGRAM_VOCAB_SIZE=0"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_v13.py 2>&1 | tee "${LOGDIR}/v13g_strip_both.log"
