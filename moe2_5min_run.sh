#!/usr/bin/env bash
set -euo pipefail

cd /workspace
mkdir -p /workspace/logs/final4_5min

run_moe() {
  local label="$1" experts="$2" layers="$3" dim="$4" mlp="$5" frac="$6"
  local logc="/workspace/logs/final4_5min/${label}.compile.log"
  local logt="/workspace/logs/final4_5min/${label}.train.log"

  local t0 t1 dt ok total_mb fit final_bpb rt_bpb

  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_8192_bpe.model DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192 \
  VOCAB_SIZE=8192 NUM_HEADS=8 NUM_KV_HEADS=8 TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 \
  VAL_BATCH_SIZE=2048 VAL_MAX_TOKENS=4096 FAST_SMOKE=1 FAST_SMOKE_BATCHES=16 SLIDING_EVAL=0 TTT_ENABLED=0 \
  DDP_FIND_UNUSED_PARAMETERS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCHINDUCTOR_COMPILE_THREADS=32 TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
  TORCHINDUCTOR_CACHE_DIR=/workspace/torch_cache TRITON_CACHE_DIR=/workspace/triton_cache \
  MOE_ENABLED=1 MOE_NUM_EXPERTS="$experts" MOE_TOP_K=1 MOE_LAYER_FRAC="$frac" NUM_LAYERS="$layers" MODEL_DIM="$dim" MLP_MULT="$mlp" SHARED_BLOCKS=0 \
  COMPILE_MODE=max-autotune COMPILER_WARMUP_STEPS=1 PRECOMPILE_ONLY=1 SYNTHETIC_WARMUP=1 ITERATIONS=0 MAX_WALLCLOCK_SECONDS=20 \
  python3 build_submission.py >/dev/null && python3 train_gpt.py >"$logc" 2>&1 || true
  t1=$(date +%s)
  dt=$((t1 - t0))
  ok="no"
  grep -q "precompile_only:done" "$logc" && ok="yes"

  CUDA_VISIBLE_DEVICES=1 TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_8192_bpe.model DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192 \
  VOCAB_SIZE=8192 NUM_HEADS=8 NUM_KV_HEADS=8 TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 \
  VAL_BATCH_SIZE=2048 VAL_MAX_TOKENS=4096 FAST_SMOKE=1 FAST_SMOKE_BATCHES=16 SLIDING_EVAL=0 TTT_ENABLED=0 \
  DDP_FIND_UNUSED_PARAMETERS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCHINDUCTOR_CACHE_DIR=/workspace/torch_cache TRITON_CACHE_DIR=/workspace/triton_cache \
  MOE_ENABLED=1 MOE_NUM_EXPERTS="$experts" MOE_TOP_K=1 MOE_LAYER_FRAC="$frac" NUM_LAYERS="$layers" MODEL_DIM="$dim" MLP_MULT="$mlp" SHARED_BLOCKS=0 \
  COMPILE_MODE=none COMPILER_WARMUP_STEPS=0 PRECOMPILE_ONLY=0 SYNTHETIC_WARMUP=0 ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=300 \
  python3 build_submission.py >/dev/null && python3 train_gpt.py >"$logt" 2>&1 || true

  total_mb=$(grep -Eo 'budget:[0-9]+/16000000 \([0-9.]+/16.00MB\)' "$logt" | tail -n1 | sed -E 's/.*\(([0-9.]+)\/16.00MB\)/\1/' || true)
  fit=$(grep -Eo '(FITS|OVER)$' "$logt" | tail -n1 || true)
  final_bpb=$(grep -Eo 'final_evaluation:completed val_loss:[0-9.]+ val_bpb:[0-9.]+' "$logt" | tail -n1 | sed -E 's/.* val_bpb:([0-9.]+)/\1/' || true)
  rt_bpb=$(grep -Eo 'final_ternary_roundtrip val_loss:[0-9.]+ val_bpb:[0-9.]+' "$logt" | tail -n1 | sed -E 's/.* val_bpb:([0-9.]+)/\1/' || true)

  echo -e "${label}\t1\t${experts}\t${layers}\t${dim}\t${mlp}\t0\t${frac}\t${dt}\t${ok}\t${total_mb}\t${fit}\t${final_bpb}\t${rt_bpb}\t${logt}" >> /workspace/logs/final4_5min/summary.tsv
  echo "done ${label} compile_s=${dt} total_mb=${total_mb} final=${final_bpb} rt=${rt_bpb}"
}

run_moe moe_a 7 9 400 4 0.50
run_moe moe_b 8 10 384 4 0.60

echo "DONE /workspace/logs/final4_5min/summary.tsv"
