#!/bin/bash
# Parameter Golf: Full 3-seed run on 8xH100
set -e
cd /workspace/parameter-golf

for SEED in 42 1337 7; do
  echo "========== SEED $SEED =========="
  RUN_ID=full_seed${SEED} \
  SEED=$SEED \
  TRAIN_SEQ_LEN=2048 \
  TRAIN_BATCH_TOKENS=524288 \
  WARMDOWN_ITERS=5000 \
  MUON_MOMENTUM=0.95 \
  MUON_MOMENTUM_WARMUP_START=0.85 \
  MUON_MOMENTUM_WARMUP_STEPS=500 \
  MATRIX_LR=0.02 \
  SCALAR_LR=0.02 \
  TIED_EMBED_LR=0.03 \
  GRAD_CLIP_NORM=1.0 \
  EMA_DECAY=0.999 \
  INT6_LAYERS="" \
  TTT_LORA_RANK=0 \
  torchrun --standalone --nproc_per_node=8 our_train_gpt.py
done

echo "========== RESULTS =========="
grep "final_int8_zlib_roundtrip_exact" logs/full_seed*.txt
