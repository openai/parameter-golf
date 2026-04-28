#!/bin/bash
# v13f: Hyperparameter tuning aligned with PR #1493
#
# Changes from v13a:
#   MUON_WD=0.095 (was 0.085)
#   MATRIX_LR=0.022 (was 0.025)
# These are the two hyperparams where PR #1493 diverges from our config.
#
# Purpose: test if PR #1493's tuned hyperparams improve our stack.
# Isolated test — no architecture changes from v13a.
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v13f_hparam_tune"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# Architecture — same as v13a baseline
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export MATRIX_CLIP_SIGMAS=12.85
export PARTIAL_KEY_OFFSET=1
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export RECUR_LAYERS="4,5"
export RECUR_START_FRAC=0.5
export WARMDOWN_ITERS=4000
export VE_ENABLED=0
export TARGET_MB=15.2

# v13f changes: PR #1493 hyperparams
export MUON_WD=0.095
export MATRIX_LR=0.022

# No TTT, no parallel
export TTT_ENABLED=0
export PARALLEL_START_LAYER=-1

echo "========================================"
echo "  v13f: Hyperparameter tuning (PR #1493)"
echo "  MUON_WD=0.095 (was 0.085)"
echo "  MATRIX_LR=0.022 (was 0.025)"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_v13.py 2>&1 | tee "${LOGDIR}/v13f_hparam_tune.log"
