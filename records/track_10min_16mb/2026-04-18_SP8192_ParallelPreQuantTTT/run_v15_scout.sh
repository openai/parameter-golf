#!/bin/bash
# V15 Scout: PR #1735 + CaseOps tokenizer
# Run with: bash records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/run_v15_scout.sh
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/

SEED=${SEED:-1337}
echo "========== V15 SCOUT SEED $SEED [$(date)] =========="

env SEED=$SEED \
  DATA_DIR=/workspace/caseops_data/datasets/ \
  DATASETS_DIR=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  TTT_EMA_ENABLED=0 \
  PREQUANT_TTT_ENABLED=1 \
  PREQUANT_TTT_EPOCHS=21 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/scout_v15_seed${SEED}.log 2>&1

echo "========== DONE [$(date)] =========="
echo "=== Final BPB ==="
grep -E "byte_sidecar|prequant_ttt:epoch 21|sliding|Total submission|val_bpb|stopping_early|final_int6|Quantized weights" /workspace/scout_v15_seed${SEED}.log | tail -30
