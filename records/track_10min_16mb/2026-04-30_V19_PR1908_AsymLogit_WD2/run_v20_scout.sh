#!/bin/bash
# V20 scout: V19c lessons applied — drop MATRIX_LR penalty, keep TTT helpers, add LORA_RANK=144
#
# V19c data analysis (single-seed 42):
#   MATRIX_LR=0.028 (vs 0.026 default) hurt pre-quant by +0.005 BPB
#   AsymLogit + PHASED_TTT_PREFIX=3500 helped TTT recovery by ~-0.002 BPB
#   Net: V19c lost -0.001 BPB vs PR #1908 frontier
#
# V20 = remove the LR penalty + keep both TTT helpers + add modest LORA_RANK bump:
#   - DROP MATRIX_LR=0.028 -> back to 0.026 default (avoid +0.005 train penalty)
#   - KEEP ASYM_LOGIT_RESCALE=1 (eval-only, V19c proved -0.001~-0.002)
#   - KEEP TTT_WEIGHT_DECAY=2.0 (stability fix, neutral on seed 42)
#   - KEEP PHASED_TTT_PREFIX_DOCS=3500 (V19c proved -0.001~-0.002, more LoRA training data)
#   - ADD TTT_LORA_RANK=144 (vs 96 default, mid-point of PR #1909's 192;
#                              50% more LoRA capacity, +20-30s eval time)
#
# Predicted (seed 42):
#   pre-quant ~1.063 (matches PR #1908 since no train hparam changes)
#   quantized ~1.072 (matches PR #1908 quant tax)
#   post-TTT ~1.057 (TTT recovery -0.013 base + AsymLogit/PHASED -0.002 + LORA_RANK -0.001 = -0.016)
#
# Win threshold: < 1.06021
# Risk: TTT_LORA_RANK=144 + PHASED_TTT_PREFIX=3500 might push eval >580s (V19c was 527s)
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/

echo "===================================================="
echo "  V20 scout: PR #1908 + AsymLogit + WD=2.0 + PHASED=3500 + LORA_RANK=144"
echo "  Seed 42  Start: $(date)"
echo "===================================================="

# CRITICAL CASEOPS_ENABLED=1 (matches PR #1908 actual training).
ENV_VARS="DATA_DIR=/workspace/caseops_data/datasets/ \
  CASEOPS_ENABLED=1 \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  ASYM_LOGIT_RESCALE=1 \
  TTT_WEIGHT_DECAY=2.0 \
  PHASED_TTT_PREFIX_DOCS=3500 \
  TTT_LORA_RANK=144 \
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
  > /workspace/scout_v20_seed42.log 2>&1

cp final_model.int6.ptz /workspace/v20_seed42_model.int6.ptz 2>/dev/null || true
cp /workspace/scout_v20_seed42.log /workspace/v20_seed42_FULL.log 2>/dev/null || true

echo ""
echo "===================================================="
echo "  V20 scout DONE  $(date)"
echo "===================================================="
grep -E "stopping_early|train_time|quantized_ttt_phased|val_bpb|total_eval_time" /workspace/scout_v20_seed42.log | tail -10
echo ""
echo "DECISION RULE:"
echo "  PR #1908 reported 3-seed mean: 1.06081"
echo "  community merge floor:         0.0006 BPB"
echo "  win threshold:                 < 1.06021"
echo ""
echo "  if V20 quantized_ttt_phased < 1.058   -> CLEAR WIN, commit pre-pay 3-seed"
echo "  if V20 quantized_ttt_phased 1.058-1.060 -> WIN, run 3-seed"
echo "  if V20 quantized_ttt_phased 1.060-1.062 -> tied, ablate or stop"
echo "  if V20 quantized_ttt_phased > 1.062   -> regression, stop"
