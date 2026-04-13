#!/usr/bin/env bash
set -euo pipefail

cd /workspace
mkdir -p /workspace/logs/final4_100s_compiled_strict
SUMMARY=/workspace/logs/final4_100s_compiled_strict/summary.tsv

echo -e "label\tmoe\texperts\tlayers\tdim\tmlp\tshared\tmoe_frac\tcompile_s\tcompile_ok\ttrain_exit\tsteps_seen\tfirst_step\tfirst_loss\tlast_step\tlast_loss\tloss_drop\tloss_drop_per_step\tlast_avg_ms_per_step\trecompile_warnings\tcache_limit_warnings\ttotal_mb\tbudget_fit\tfinal_val_bpb\troundtrip_bpb\tlog_compile\tlog_train" > "$SUMMARY"

run_one() {
  local label="$1" moe="$2" experts="$3" layers="$4" dim="$5" mlp="$6" shared="$7" frac="$8"
  local logc="/workspace/logs/final4_100s_compiled_strict/${label}.compile.log"
  local logt="/workspace/logs/final4_100s_compiled_strict/${label}.train.log"

  local t0 t1 dt ok train_exit
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_8192_bpe.model DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192 \
  VOCAB_SIZE=8192 NUM_HEADS=8 NUM_KV_HEADS=8 TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 \
  VAL_BATCH_SIZE=2048 VAL_MAX_TOKENS=4096 FAST_SMOKE=1 FAST_SMOKE_BATCHES=16 \
  SLIDING_EVAL=0 TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 \
  DDP_FIND_UNUSED_PARAMETERS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCHINDUCTOR_COMPILE_THREADS=32 TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
  TORCHINDUCTOR_CACHE_DIR=/workspace/torch_cache TRITON_CACHE_DIR=/workspace/triton_cache \
  TORCHDYNAMO_CACHE_SIZE_LIMIT=256 \
  MOE_ENABLED="$moe" MOE_NUM_EXPERTS="$experts" MOE_TOP_K=1 MOE_LAYER_FRAC="$frac" \
  NUM_LAYERS="$layers" MODEL_DIM="$dim" MLP_MULT="$mlp" SHARED_BLOCKS="$shared" \
  COMPILE_MODE=max-autotune COMPILER_WARMUP_STEPS=1 PRECOMPILE_ONLY=1 SYNTHETIC_WARMUP=1 \
  ITERATIONS=0 MAX_WALLCLOCK_SECONDS=20 \
  python3 build_submission.py >/dev/null && python3 train_gpt.py >"$logc" 2>&1 || true
  t1=$(date +%s)
  dt=$((t1 - t0))
  ok="no"
  grep -q "precompile_only:done" "$logc" && ok="yes"

  set +e
  timeout 900s env \
    CUDA_VISIBLE_DEVICES=1 TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_8192_bpe.model DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192 \
    VOCAB_SIZE=8192 NUM_HEADS=8 NUM_KV_HEADS=8 TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 \
    VAL_BATCH_SIZE=2048 VAL_MAX_TOKENS=4096 FAST_SMOKE=1 FAST_SMOKE_BATCHES=16 \
    SLIDING_EVAL=0 TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 \
    DDP_FIND_UNUSED_PARAMETERS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCHINDUCTOR_COMPILE_THREADS=32 TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
    TORCHINDUCTOR_CACHE_DIR=/workspace/torch_cache TRITON_CACHE_DIR=/workspace/triton_cache \
    TORCHDYNAMO_CACHE_SIZE_LIMIT=256 \
    MOE_ENABLED="$moe" MOE_NUM_EXPERTS="$experts" MOE_TOP_K=1 MOE_LAYER_FRAC="$frac" \
    NUM_LAYERS="$layers" MODEL_DIM="$dim" MLP_MULT="$mlp" SHARED_BLOCKS="$shared" \
    COMPILE_MODE=max-autotune COMPILER_WARMUP_STEPS=0 PRECOMPILE_ONLY=0 SYNTHETIC_WARMUP=0 \
    ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=100 TRAIN_LOG_EVERY=50 TRAIN_LOG_EVERY_FRACTION=0 \
    python3 build_submission.py >/dev/null && python3 train_gpt.py >"$logt" 2>&1
  train_exit=$?
  set -e

  python3 - "$SUMMARY" "$label" "$moe" "$experts" "$layers" "$dim" "$mlp" "$shared" "$frac" "$dt" "$ok" "$train_exit" "$logc" "$logt" <<'PY'
import re, sys
summary, label, moe, experts, layers, dim, mlp, shared, frac, compile_s, compile_ok, train_exit, logc, logt = sys.argv[1:]
step_re = re.compile(r"step:(\d+)/\d+ loss:([0-9.]+) t:[0-9.]+ms avg:([0-9.]+)ms")
final_re = re.compile(r"final_evaluation:completed val_loss:[0-9.]+ val_bpb:([0-9.]+)")
rt_re = re.compile(r"final_ternary_roundtrip val_loss:[0-9.]+ val_bpb:([0-9.]+)")
budget_re = re.compile(r"budget:[0-9]+/16000000 \(([0-9.]+)/16.00MB\) (FITS|OVER)")

steps = []
final_bpb = ""
rt_bpb = ""
total_mb = ""
fit = ""
recompile_warnings = 0
cache_limit_warnings = 0
with open(logt, "r", errors="ignore") as f:
    for line in f:
        m = step_re.search(line)
        if m:
            steps.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        m2 = final_re.search(line)
        if m2:
            final_bpb = m2.group(1)
        m3 = rt_re.search(line)
        if m3:
            rt_bpb = m3.group(1)
        m4 = budget_re.search(line)
        if m4:
            total_mb, fit = m4.group(1), m4.group(2)
        if "recompil" in line.lower():
            recompile_warnings += 1
        if "cache_size_limit" in line:
            cache_limit_warnings += 1

if steps:
    first_step, first_loss, _ = steps[0]
    last_step, last_loss, last_avg_ms = steps[-1]
    span = max(last_step - first_step, 1)
    loss_drop = first_loss - last_loss
    loss_drop_per_step = loss_drop / span
    steps_seen = len(steps)
else:
    steps_seen = 0
    first_step = last_step = 0
    first_loss = last_loss = last_avg_ms = loss_drop = loss_drop_per_step = 0.0

row = [
    label, moe, experts, layers, dim, mlp, shared, frac, compile_s, compile_ok, train_exit,
    str(steps_seen), str(first_step), f"{first_loss:.6f}",
    str(last_step), f"{last_loss:.6f}", f"{loss_drop:.6f}",
    f"{loss_drop_per_step:.8f}", f"{last_avg_ms:.2f}",
    str(recompile_warnings), str(cache_limit_warnings),
    total_mb, fit, final_bpb, rt_bpb, logc, logt
]
with open(summary, "a") as out:
    out.write("\t".join(row) + "\n")
print("done", label, "compile_s", compile_s, "train_exit", train_exit, "steps", steps_seen)
PY
}

run_one nm_a 0 1 16 456 5 2 0.67
run_one nm_b 0 1 16 448 5 2 0.67
run_one moe_a 1 7 9 400 4 0 0.50
run_one moe_b 1 8 10 384 4 0 0.60

echo "DONE $SUMMARY"
