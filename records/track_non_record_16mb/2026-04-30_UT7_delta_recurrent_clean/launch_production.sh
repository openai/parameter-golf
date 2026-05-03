#!/bin/bash
# Full 8xH100 non-record launcher. Default runs seed 42 only.
# After seed 42 passes, run: SEEDS="314" bash launch_production.sh

set -euo pipefail

cd /workspace/parameter-golf/records/track_non_record_16mb/2026-04-30_UT7_delta_recurrent_clean

export VOCAB_SIZE=8192
export DATA_PATH=../../../data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=../../../data/tokenizers/fineweb_8192_bpe.model

export MODEL_DIM=1024 D_FF=3072 NUM_HEADS=8 NUM_KV_HEADS=4 HEAD_DIM=128
export ADAPTER_RANK=320 K_ITERS=6 TTT_CHUNK_SIZE=32
export USE_RLMA=1 USE_TTT=0 LOG_ETA_INIT=-2.0 GRAD_CHECKPOINT=0
export UT_RESIDUAL_DELTA=1 BRANCH_SCALE_INIT=0.6

export ITERATIONS=2400 WARMUP_STEPS=50 WARMDOWN_ITERS=600
export TRAIN_SEQ_LEN=8192 TRAIN_BATCH_TOKENS=262144 GRAD_ACCUM_STEPS=1
export MATRIX_LR=0.026 GRAD_CLIP_NORM=0.2
export VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=25 MAX_WALLCLOCK_SECONDS=0
export EVAL_CTX=8192 EVAL_STRIDE=8192 SW_ATTN_WINDOW_EVAL=512 VAL_TOKEN_LIMIT=0
export GPTQ_BITS=8 GPTQ_CLIP_K=12.85 EMBED_QUANT_BITS=8 ZSTD_LEVEL=22
export TARGET_ARTIFACT_BYTES=16000000 FINAL_FP_EVAL=${FINAL_FP_EVAL:-1} QUANT_SWEEP_SPECS=""

mkdir -p artifacts

for SEED in ${SEEDS:-42}; do
  export SEED
  export RUN_ID="prod_seed${SEED}"
  echo "=== seed=${SEED} start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "artifacts/prod_seed${SEED}.console.log"
  cp "logs/prod_seed${SEED}.txt" "artifacts/train_seed${SEED}.log"
  if [ -f final_model.rlma_int6_int8.zst ]; then
    cp final_model.rlma_int6_int8.zst "artifacts/final_model_seed${SEED}.zst"
    echo "artifact_seed${SEED}_bytes=$(stat -c%s artifacts/final_model_seed${SEED}.zst)"
  fi
  grep -E 'lr_scale:0\.' "artifacts/train_seed${SEED}.log" >/dev/null || {
    echo "seed ${SEED} failed: warmdown did not appear in log" >&2
    exit 1
  }
  grep "final_quant_roundtrip_exact" "artifacts/train_seed${SEED}.log"
  echo "=== seed=${SEED} end $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done
