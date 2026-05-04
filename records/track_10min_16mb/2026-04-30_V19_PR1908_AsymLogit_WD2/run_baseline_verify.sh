#!/bin/bash
# CRITICAL VERIFICATION: reproduce PR #1908's reported 1.05957 (seed 42 alone)
# with CASEOPS_ENABLED=1 and FORCE_STOP_STEP=4945 matching their submission.
#
# If this gives val_bpb ~1.0596, our setup matches PR #1908's eval pipeline.
# If it gives 0.97 again, CASEOPS_ENABLED isn't taking effect for some reason.
# If it gives 1.05-1.07 but not 1.0596, our dataset shards differ from theirs.
#
# RUN THIS FIRST. ~19 min, ~$0.65.
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/

echo "===================================================="
echo "  BASELINE VERIFY: PR #1908 unchanged + CASEOPS_ENABLED=1"
echo "  Seed 42, FORCE_STOP_STEP=4945  Start: $(date)"
echo "===================================================="

# PR #1908's exact reported env vars from their record README
# NO V19 changes. NO simon-marcus changes. NO TTT_WD override.
ENV_VARS="DATA_DIR=/workspace/caseops_data/datasets/ \
  CASEOPS_ENABLED=1 \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  FORCE_STOP_STEP=4945 \
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
  > /workspace/baseline_verify_seed42.log 2>&1

cp final_model.int6.ptz /workspace/baseline_verify_seed42_model.int6.ptz 2>/dev/null || true

echo ""
echo "===================================================="
echo "  BASELINE VERIFY DONE  $(date)"
echo "===================================================="
grep -E "caseops_enabled|stopping_early|train_time|quantized_ttt_phased|val_bpb" /workspace/baseline_verify_seed42.log | tail -10
echo ""
echo "EXPECTED: val_bpb ~1.05957 (matches PR #1908 seed 42 reported)"
echo ""
echo "If output shows:"
echo "  caseops_enabled: True   AND   val_bpb in 1.058-1.061 range"
echo "  -> setup correct, proceed to V19c scout"
echo ""
echo "  caseops_enabled: False  OR    val_bpb ~0.97"
echo "  -> CASEOPS_ENABLED not taking effect, debug needed"
