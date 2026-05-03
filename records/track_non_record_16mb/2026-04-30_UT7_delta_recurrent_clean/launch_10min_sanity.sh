#!/bin/bash
# 8xH100 sanity seed before production. Uses the locked UT7 rank320/q8 shape.

set -euo pipefail

cd /workspace/parameter-golf/records/track_non_record_16mb/2026-04-30_UT7_delta_recurrent_clean

export RUN_ID=${RUN_ID:-ut7_rank320_q8_8x_sanity_s42}
export SEED=${SEED:-42}
export VOCAB_SIZE=8192
export DATA_PATH=../../../data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=../../../data/tokenizers/fineweb_8192_bpe.model

export MODEL_DIM=1024 D_FF=3072 NUM_HEADS=8 NUM_KV_HEADS=4 HEAD_DIM=128
export ADAPTER_RANK=320 K_ITERS=6 TTT_CHUNK_SIZE=32
export USE_RLMA=1 USE_TTT=0 LOG_ETA_INIT=-2.0 GRAD_CHECKPOINT=0
export UT_RESIDUAL_DELTA=1 BRANCH_SCALE_INIT=0.6

export ITERATIONS=240 WARMUP_STEPS=25 WARMDOWN_ITERS=60
export TRAIN_SEQ_LEN=8192 TRAIN_BATCH_TOKENS=262144 GRAD_ACCUM_STEPS=1
export MATRIX_LR=0.026 GRAD_CLIP_NORM=0.2
export VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=20 MAX_WALLCLOCK_SECONDS=0
export EVAL_CTX=8192 EVAL_STRIDE=8192 SW_ATTN_WINDOW_EVAL=512 VAL_TOKEN_LIMIT=524288
export GPTQ_BITS=8 GPTQ_CLIP_K=12.85 EMBED_QUANT_BITS=8 ZSTD_LEVEL=22
export TARGET_ARTIFACT_BYTES=16000000 FINAL_FP_EVAL=1 QUANT_SWEEP_SPECS=""

mkdir -p artifacts_sanity
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "artifacts_sanity/${RUN_ID}.console.log"
cp "logs/${RUN_ID}.txt" "artifacts_sanity/${RUN_ID}.log"
if [ -f final_model.rlma_int6_int8.zst ]; then
  cp final_model.rlma_int6_int8.zst "artifacts_sanity/${RUN_ID}.zst"
fi

grep -E 'lr_scale:0\.' "artifacts_sanity/${RUN_ID}.log" >/dev/null || {
  echo "sanity failed: warmdown did not appear in log" >&2
  exit 1
}
grep "final_quant_roundtrip_exact" "artifacts_sanity/${RUN_ID}.log"
