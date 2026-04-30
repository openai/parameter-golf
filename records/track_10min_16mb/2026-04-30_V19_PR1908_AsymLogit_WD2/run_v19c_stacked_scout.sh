#!/bin/bash
# V19c FULL STACK scout: PR #1908 + Asymmetric Logit Rescale + simon-marcus hparams
# Single seed 42, ~19 min, ~$0.65.
#
# Combines THREE independent improvements (each verified separately by community):
#   1. Asymmetric Logit Rescale (PR #1923 jorge-asenjo)
#      - sunnypatneedi flagged "empirical negative" but ONLY on PR #1855 base
#        with WD=1.0 default. Never tested on PR #1908 + WD=2.0.
#   2. simon-marcus hparams (PR #1925, 3-seed verified 1.06049 on PR #1855 base)
#      - MATRIX_LR 0.026 -> 0.028
#      - PHASED_TTT_PREFIX_DOCS 2500 -> 3500
#   3. TTT_WEIGHT_DECAY 1.0 -> 2.0 (PR #1886 fused-CE collapse fix)
#
# Theory: 3 orthogonal axes; if any 1 wins, we beat PR #1908 frontier.
# If V19c regresses, we can ablate (run V19a alone first, or V19b separately).
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/

echo "===================================================="
echo "  V19c STACKED scout: PR #1908 + 3 axes"
echo "  Seed 42  Start: $(date)"
echo "===================================================="

# CRITICAL: CASEOPS_ENABLED=1 + explicit DATA_PATH/TOKENIZER_PATH so BPB
# accounting uses the byte sidecar (fineweb_val_bytes_*.bin) — matches
# PR #1908's actual training log (caseops_enabled: True). Without this
# the code falls back to SP LUT byte counting → BPB ~0.97 instead of ~1.06.
ENV_VARS="DATA_DIR=/workspace/caseops_data/datasets/ \
  CASEOPS_ENABLED=1 \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  ASYM_LOGIT_RESCALE=1 \
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
  > /workspace/scout_v19c_seed42.log 2>&1

cp final_model.int6.ptz /workspace/v19c_seed42_model.int6.ptz 2>/dev/null || true
cp /workspace/scout_v19c_seed42.log /workspace/v19c_seed42_FULL.log 2>/dev/null || true

echo ""
echo "===================================================="
echo "  V19c scout DONE  $(date)"
echo "===================================================="
grep -E "stopping_early|train_time|quantized_ttt_phased|val_bpb" /workspace/scout_v19c_seed42.log | tail -10
echo ""
echo "DECISION RULE (with CASEOPS_ENABLED=1, byte sidecar BPB):"
echo "  PR #1908 reported 3-seed mean: 1.06081"
echo "  community merge floor:         0.0006 BPB"
echo "  win threshold:                 < 1.06021"
echo ""
echo "  if V19c < 1.06021 -> CLEAR WIN (>floor), run 3-seed"
echo "  if V19c 1.06021-1.0608 -> borderline, ablate (V19a/V19b)"
echo "  if V19c > 1.0608 -> regression, fallback to Lead B"
