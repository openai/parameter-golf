#!/usr/bin/env bash
set -euo pipefail

cd /workspace
mkdir -p /workspace/logs/moe_strict_100s_only
SUMMARY=/workspace/logs/moe_strict_100s_only/summary.tsv
echo -e "label\texperts\tlayers\tdim\tmoe_frac\tcompile_s\tcompile_ok\ttrain_exit\tsteps_seen\tfirst_step\tfirst_loss\tlast_step\tlast_loss\tloss_drop\tloss_drop_per_step\tlast_avg_ms_per_step\trecompile_warnings\tcache_limit_warnings\tfinal_val_bpb\troundtrip_bpb\tlog_compile\tlog_train" > "$SUMMARY"

run_one() {
  local label="$1" experts="$2" layers="$3" dim="$4" frac="$5"
  local logc="/workspace/logs/moe_strict_100s_only/${label}.compile.log"
  local logt="/workspace/logs/moe_strict_100s_only/${label}.train.log"

  local t0 t1 dt ok train_exit
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_8192_bpe.model DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192 \
  VOCAB_SIZE=8192 NUM_HEADS=8 NUM_KV_HEADS=8 TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 \
  VAL_BATCH_SIZE=2048 VAL_MAX_TOKENS=4096 FAST_SMOKE=1 FAST_SMOKE_BATCHES=16 SLIDING_EVAL=0 TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 \
  DDP_FIND_UNUSED_PARAMETERS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_COMPILE_THREADS=32 \
  TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 TORCHINDUCTOR_CACHE_DIR=/workspace/torch_cache TRITON_CACHE_DIR=/workspace/triton_cache \
  TORCHDYNAMO_CACHE_SIZE_LIMIT=256 \
  MOE_ENABLED=1 MOE_NUM_EXPERTS="$experts" MOE_TOP_K=1 MOE_LAYER_FRAC="$frac" NUM_LAYERS="$layers" MODEL_DIM="$dim" MLP_MULT=4 SHARED_BLOCKS=0 \
  COMPILE_MODE=max-autotune COMPILER_WARMUP_STEPS=1 PRECOMPILE_ONLY=1 SYNTHETIC_WARMUP=1 ITERATIONS=0 MAX_WALLCLOCK_SECONDS=20 \
  python3 build_submission.py >/dev/null && python3 train_gpt.py >"$logc" 2>&1 || true
  t1=$(date +%s); dt=$((t1 - t0)); ok=no
  grep -q "precompile_only:done" "$logc" && ok=yes

  set +e
  timeout 900s env \
    CUDA_VISIBLE_DEVICES=1 TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_8192_bpe.model DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192 \
    VOCAB_SIZE=8192 NUM_HEADS=8 NUM_KV_HEADS=8 TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 \
    VAL_BATCH_SIZE=2048 VAL_MAX_TOKENS=4096 FAST_SMOKE=1 FAST_SMOKE_BATCHES=16 SLIDING_EVAL=0 TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 \
    DDP_FIND_UNUSED_PARAMETERS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_COMPILE_THREADS=32 \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 TORCHINDUCTOR_CACHE_DIR=/workspace/torch_cache TRITON_CACHE_DIR=/workspace/triton_cache \
    TORCHDYNAMO_CACHE_SIZE_LIMIT=256 \
    MOE_ENABLED=1 MOE_NUM_EXPERTS="$experts" MOE_TOP_K=1 MOE_LAYER_FRAC="$frac" NUM_LAYERS="$layers" MODEL_DIM="$dim" MLP_MULT=4 SHARED_BLOCKS=0 \
    COMPILE_MODE=max-autotune COMPILER_WARMUP_STEPS=0 PRECOMPILE_ONLY=0 SYNTHETIC_WARMUP=0 ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=100 \
    TRAIN_LOG_EVERY=1 TRAIN_LOG_EVERY_FRACTION=0 \
    python3 build_submission.py >/dev/null && python3 train_gpt.py >"$logt" 2>&1
  train_exit=$?
  set -e

  python3 - "$SUMMARY" "$label" "$experts" "$layers" "$dim" "$frac" "$dt" "$ok" "$train_exit" "$logc" "$logt" <<'PY'
import re, sys
summary, label, experts, layers, dim, frac, compile_s, compile_ok, train_exit, logc, logt = sys.argv[1:]
step_re = re.compile(r"step:(\d+)/\d+ loss:([0-9.]+) t:[0-9.]+ms avg:([0-9.]+)ms")
final_re = re.compile(r"final_evaluation:completed val_loss:[0-9.]+ val_bpb:([0-9.]+)")
rt_re = re.compile(r"final_ternary_roundtrip val_loss:[0-9.]+ val_bpb:([0-9.]+)")
steps=[]; final_bpb=""; rt_bpb=""; recomp=0; cache=0
with open(logt, "r", errors="ignore") as f:
    for line in f:
        m=step_re.search(line)
        if m: steps.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        m2=final_re.search(line)
        if m2: final_bpb=m2.group(1)
        m3=rt_re.search(line)
        if m3: rt_bpb=m3.group(1)
        if "recompil" in line.lower(): recomp += 1
        if "cache_size_limit" in line: cache += 1
if steps:
    fs, fl, _ = steps[0]; ls, ll, avg = steps[-1]
    span=max(ls-fs,1); drop=fl-ll; dps=drop/span; n=len(steps)
else:
    fs=ls=n=0; fl=ll=avg=drop=dps=0.0
row=[label,experts,layers,dim,frac,compile_s,compile_ok,train_exit,str(n),str(fs),f"{fl:.6f}",str(ls),f"{ll:.6f}",f"{drop:.6f}",f"{dps:.8f}",f"{avg:.2f}",str(recomp),str(cache),final_bpb,rt_bpb,logc,logt]
with open(summary,"a") as o: o.write("\t".join(map(str,row))+"\n")
print("done",label,"steps",n,"train_exit",train_exit,"recompile",recomp,"cache",cache)
PY
}

run_one moe_a 7 9 400 0.50
run_one moe_b 8 10 384 0.60
