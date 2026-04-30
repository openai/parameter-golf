#!/bin/bash
# V19 scout: single seed 42 on PR #1908 base + Asymmetric Logit Rescale + TTT_WD=2.0
# Expected runtime: ~12 min train + ~7 min eval = ~19 min total
# Cost on 8xH100 SXM @ ~$2/hr: ~$0.65
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/

echo "===================================================="
echo "  V19 scout: PR #1908 + AsymLogit + TTT_WD=2.0"
echo "  Seed 42  Start: $(date)"
echo "===================================================="

# Inherits PR #1908 stack:
#   AWQ_LITE (8 bits, 1 group, 64 cols) + LQER asym int4 rank-4
#   Phased TTT (prefix=2500) + sparse_attn_gate + BOS-fixed SmearGate
# V19 additions (env vars only):
#   ASYM_LOGIT_RESCALE=1     (turn on PR #1923 asymmetric softcap)
#   TTT_WEIGHT_DECAY=2.0     (PR #1886 fused-CE stability fix; default in train_gpt.py)
# CRITICAL: CASEOPS_ENABLED=1 makes the code load the byte sidecar
# (fineweb_val_bytes_*.bin) for BPB accounting. Without this flag the code
# falls back to SentencePiece LUT byte-counting which gives ~0.97 BPB instead
# of the correct ~1.06 BPB. PR #1908's training log shows caseops_enabled: True.
ENV_VARS="DATA_DIR=/workspace/caseops_data/datasets/ \
  CASEOPS_ENABLED=1 \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  ASYM_LOGIT_RESCALE=1 \
  TTT_WEIGHT_DECAY=2.0 \
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
  > /workspace/scout_v19_seed42.log 2>&1

cp final_model.int6.ptz /workspace/v19_seed42_model.int6.ptz 2>/dev/null || true
cp /workspace/scout_v19_seed42.log /workspace/v19_seed42_FULL.log 2>/dev/null || true

echo ""
echo "===================================================="
echo "  V19 scout DONE  $(date)"
echo "===================================================="
grep -E "stopping_early|train_time|quantized_ttt_phased|val_bpb" /workspace/scout_v19_seed42.log | tail -10
echo ""
echo "DECISION RULE (NEW with CASEOPS_ENABLED=1):"
echo "  PR #1908 reported (3-seed mean):  1.06081"
echo "  community merge floor:            0.0006 BPB"
echo "  win threshold:                    < 1.06021"
echo ""
echo "  if V19 quantized_ttt_phased < 1.06021 -> TRUE WIN, run run_v19_3seeds.sh"
echo "  if V19 quantized_ttt_phased 1.06021-1.0608 -> borderline, ablate"
echo "  if V19 quantized_ttt_phased > 1.0608 -> regression"
