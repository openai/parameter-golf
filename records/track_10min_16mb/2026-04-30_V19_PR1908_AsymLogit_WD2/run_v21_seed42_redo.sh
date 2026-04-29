#!/bin/bash
# V21 seed 42 REDO — strict <600s wallclock per @aquariouseworkman + @romeerp review
# Same config as V21 seeds 0 + 1234 (GPTQ_RESERVE=4.0, no FORCE_STOP_STEP)
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/

echo "===================================================="
echo "  V21 SEED 42 REDO (strict <600s)  Start: $(date)"
echo "===================================================="

env SEED=42 \
  DATA_DIR=/workspace/caseops_data/datasets/ \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
  ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
  WARMUP_STEPS=20 WARMDOWN_FRAC=0.85 BETA2=0.99 \
  GRAD_CLIP_NORM=0.3 MIN_LR=0.1 MATRIX_LR=0.026 \
  GLOBAL_TTT_MOMENTUM=0.9 \
  SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 GATED_ATTN_QUANT_GATE=1 \
  FUSED_CE_ENABLED=1 EMBED_BITS=7 \
  MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
  GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 COMPRESSOR=pergroup \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 \
  LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
  AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
  TTT_CHUNK_SIZE=48 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
  MUON_BACKEND_STEPS=5 NCCL_NET=Socket VAL_LOSS_EVERY=0 \
  ASYM_LOGIT_RESCALE=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/scout_v21_seed42_REDO.log 2>&1

cp final_model.int6.ptz /workspace/v21_seed42_REDO_model.int6.ptz 2>/dev/null || true

echo ""
echo "===================================================="
echo "  V21 SEED 42 REDO DONE  $(date)"
echo "===================================================="
grep -E "stopping_early|train_time|quantized_ttt_phased|Total submission|total_eval_time" /workspace/scout_v21_seed42_REDO.log | tail -8
