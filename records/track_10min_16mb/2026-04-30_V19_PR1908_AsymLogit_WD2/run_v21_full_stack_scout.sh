#!/bin/bash
# V21 = FULL PR #1855 9-hp stack + PR #1908 AWQ-lite + V19 ASYM_LOGIT_RESCALE
# This is the FIRST version with the COMPLETE PR #1855 reproduction env vars.
# V18/V19c/V20 all ran with SmearGate=False, SparseAttnGate=False, num_phases=1 -> WRONG BASE.
# Source: PR #1855 README lines 125-145 (codemath3000's exact reproduction command).
#
# Predicted (seed 42, FORCE_STOP_STEP=4945 for direct PR #1908 comparison):
#   pre-quant val_bpb: ~1.064 (matching PR #1908 1.06384)
#   quantized val_bpb: ~1.072 (matching PR #1908 1.07226)
#   artifact size: ~15.99 MB (lrzip pergroup compression)
#   post-TTT val_bpb: ~1.057 (PR #1908 1.05957 - 0.002 from AsymLogit)
#   total eval time: ~485s (3-phase TTT slightly slower than 1-phase)
#
# Win threshold: < 1.06021
# Probability of true single-seed win vs frontier: 50-60%
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/

echo "===================================================="
echo "  V21 scout: FULL PR #1855 stack + AWQ-lite + AsymLogit"
echo "  Seed 42 + FORCE_STOP_STEP=4945  Start: $(date)"
echo "===================================================="

# COMPLETE env var set from PR #1855 README + PR #1908 AWQ-lite + V19 ASYM_LOGIT_RESCALE
ENV_VARS="DATA_DIR=/workspace/caseops_data/datasets/ \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 \
  VOCAB_SIZE=8192 \
  ITERATIONS=20000 \
  MAX_WALLCLOCK_SECONDS=600 \
  WARMUP_STEPS=20 \
  WARMDOWN_FRAC=0.85 \
  BETA2=0.99 \
  GRAD_CLIP_NORM=0.3 \
  MIN_LR=0.1 \
  MATRIX_LR=0.026 \
  GLOBAL_TTT_MOMENTUM=0.9 \
  SPARSE_ATTN_GATE_ENABLED=1 \
  SPARSE_ATTN_GATE_SCALE=0.5 \
  SMEAR_GATE_ENABLED=1 \
  GATE_WINDOW=12 \
  GATED_ATTN_QUANT_GATE=1 \
  FUSED_CE_ENABLED=1 \
  EMBED_BITS=7 \
  MLP_CLIP_SIGMAS=11.5 \
  ATTN_CLIP_SIGMAS=13.0 \
  EMBED_CLIP_SIGMAS=14.0 \
  GPTQ_RESERVE_SECONDS=0.5 \
  GPTQ_CALIBRATION_BATCHES=16 \
  COMPRESSOR=pergroup \
  LQER_ENABLED=1 \
  LQER_ASYM_ENABLED=1 \
  LQER_RANK=4 \
  LQER_FACTOR_BITS=4 \
  LQER_ASYM_GROUP=64 \
  LQER_TOP_K=3 \
  AWQ_LITE_ENABLED=1 \
  AWQ_LITE_BITS=8 \
  AWQ_LITE_GROUP_TOP_K=1 \
  AWQ_LITE_GROUP_SIZE=64 \
  PHASED_TTT_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=2500 \
  PHASED_TTT_NUM_PHASES=3 \
  TTT_CHUNK_SIZE=48 \
  TTT_BETA2=0.99 \
  TTT_WEIGHT_DECAY=0.5 \
  TTT_LORA_RANK=80 \
  MUON_BACKEND_STEPS=5 \
  NCCL_NET=Socket \
  VAL_LOSS_EVERY=0 \
  ASYM_LOGIT_RESCALE=1 \
  FORCE_STOP_STEP=4945"

env SEED=42 $ENV_VARS \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/scout_v21_seed42.log 2>&1

cp final_model.int6.ptz /workspace/v21_seed42_model.int6.ptz 2>/dev/null || true
cp /workspace/scout_v21_seed42.log /workspace/v21_seed42_FULL.log 2>/dev/null || true

echo ""
echo "===================================================="
echo "  V21 scout DONE  $(date)"
echo "===================================================="
grep -E "stopping_early|train_time|quantized_ttt_phased|val_bpb|total_eval_time|Total submission|smear_gate_enabled|sparse_attn_gate_enabled|num_phases|compressor" /workspace/scout_v21_seed42.log | tail -20
echo ""
echo "DECISION RULE:"
echo "  PR #1908 reported 3-seed mean: 1.06081"
echo "  community merge floor:         0.0006 BPB"
echo "  win threshold:                 < 1.06021"
echo "  artifact cap:                  < 16,000,000 bytes"
echo ""
echo "  if V21 quantized_ttt_phased < 1.058 AND artifact < 16M -> CLEAR WIN, run 3-seed"
echo "  if V21 quantized_ttt_phased 1.058-1.060 -> WIN, run 3-seed"
echo "  if artifact > 16M -> SIZE FAIL (debug compressor)"
echo "  if quantized_ttt_phased > 1.062 -> abandon"
