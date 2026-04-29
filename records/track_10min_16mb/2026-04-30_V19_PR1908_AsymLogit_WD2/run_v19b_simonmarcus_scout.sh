#!/bin/bash
# V19b ABLATION scout: PR #1908 + simon-marcus hparams ONLY (no AsymLogit)
# Used to ablate which axis contributed if V19c shows a partial win.
# Seed 42, ~19 min, ~$0.65.
#
# Tests JUST simon-marcus's PR #1925 deltas:
#   - MATRIX_LR 0.026 -> 0.028
#   - PHASED_TTT_PREFIX_DOCS 2500 -> 3500
#   - TTT_WD=2.0 (PR #1886 stability fix)
#
# AsymLogit is OFF (ASYM_LOGIT_RESCALE=0 default in train_gpt.py).
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/

echo "===================================================="
echo "  V19b ABLATION: PR #1908 + simon-marcus hparams"
echo "  Seed 42  Start: $(date)"
echo "===================================================="

# CRITICAL CASEOPS_ENABLED=1 (matches PR #1908 actual training run)
ENV_VARS="DATA_DIR=/workspace/caseops_data/datasets/ \
  CASEOPS_ENABLED=1 \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  TTT_WEIGHT_DECAY=2.0 \
  MATRIX_LR=0.028 \
  PHASED_TTT_PREFIX_DOCS=3500 \
  AWQ_LITE_ENABLED=1 \
  AWQ_LITE_BITS=8 \
  AWQ_LITE_GROUP_TOP_K=1 \
  AWQ_LITE_GROUP_SIZE=64 \
  LQER_ENABLED=1 \
  LQER_ASYM_ENABLED=1 \
  LQER_RANK=4 \
  LQER_FACTOR_BITS=4 \
  LQER_ASYM_GROUP=64 \
  LQER_TOP_K=3"

env SEED=42 $ENV_VARS \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/scout_v19b_seed42.log 2>&1

cp final_model.int6.ptz /workspace/v19b_seed42_model.int6.ptz 2>/dev/null || true
cp /workspace/scout_v19b_seed42.log /workspace/v19b_seed42_FULL.log 2>/dev/null || true

echo ""
echo "===================================================="
echo "  V19b scout DONE  $(date)"
echo "===================================================="
grep -E "stopping_early|train_time|quantized_ttt_phased|val_bpb" /workspace/scout_v19b_seed42.log | tail -10
