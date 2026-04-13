#!/usr/bin/env bash
set -euo pipefail

cd /workspace
mkdir -p /workspace/logs/final4_100s

SUMMARY=/workspace/logs/final4_100s/summary.tsv
echo -e "label\tmoe\texperts\tlayers\tdim\tmlp\tshared\tmoe_frac\tsteps_seen\tfirst_step\tfirst_loss\tlast_step\tlast_loss\tloss_drop\tloss_drop_per_step\tlast_avg_ms_per_step\tfinal_val_bpb\troundtrip_bpb\tlog_file" > "$SUMMARY"

run_one() {
  local label="$1" moe="$2" experts="$3" layers="$4" dim="$5" mlp="$6" shared="$7" frac="$8"
  local logf="/workspace/logs/final4_100s/${label}.train.log"

  CUDA_VISIBLE_DEVICES=1 TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_8192_bpe.model DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192 \
  VOCAB_SIZE=8192 NUM_HEADS=8 NUM_KV_HEADS=8 \
  TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 \
  VAL_BATCH_SIZE=2048 VAL_MAX_TOKENS=4096 \
  FAST_SMOKE=1 FAST_SMOKE_BATCHES=16 \
  SLIDING_EVAL=0 TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 \
  DDP_FIND_UNUSED_PARAMETERS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MOE_ENABLED="$moe" MOE_NUM_EXPERTS="$experts" MOE_TOP_K=1 MOE_LAYER_FRAC="$frac" \
  NUM_LAYERS="$layers" MODEL_DIM="$dim" MLP_MULT="$mlp" SHARED_BLOCKS="$shared" \
  COMPILE_MODE=none COMPILER_WARMUP_STEPS=0 PRECOMPILE_ONLY=0 SYNTHETIC_WARMUP=0 \
  ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=100 \
  TRAIN_LOG_EVERY=50 TRAIN_LOG_EVERY_FRACTION=0 \
  python3 build_submission.py >/dev/null && python3 train_gpt.py >"$logf" 2>&1 || true

  python3 - "$SUMMARY" "$label" "$moe" "$experts" "$layers" "$dim" "$mlp" "$shared" "$frac" "$logf" <<'PY'
import re, sys
summary, label, moe, experts, layers, dim, mlp, shared, frac, logf = sys.argv[1:]
step_re = re.compile(r"step:(\d+)/\d+ loss:([0-9.]+) t:[0-9.]+ms avg:([0-9.]+)ms")
final_re = re.compile(r"final_evaluation:completed val_loss:[0-9.]+ val_bpb:([0-9.]+)")
rt_re = re.compile(r"final_ternary_roundtrip val_loss:[0-9.]+ val_bpb:([0-9.]+)")

steps = []
final_bpb = ""
rt_bpb = ""
with open(logf, "r", errors="ignore") as f:
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

if steps:
    first_step, first_loss, _ = steps[0]
    last_step, last_loss, last_avg_ms = steps[-1]
    step_span = max(last_step - first_step, 1)
    loss_drop = first_loss - last_loss
    loss_drop_per_step = loss_drop / step_span
    steps_seen = len(steps)
else:
    first_step = last_step = steps_seen = 0
    first_loss = last_loss = loss_drop = loss_drop_per_step = last_avg_ms = 0.0

row = [
    label, moe, experts, layers, dim, mlp, shared, frac,
    str(steps_seen), str(first_step), f"{first_loss:.6f}",
    str(last_step), f"{last_loss:.6f}", f"{loss_drop:.6f}",
    f"{loss_drop_per_step:.8f}", f"{last_avg_ms:.2f}",
    final_bpb, rt_bpb, logf
]
with open(summary, "a") as out:
    out.write("\t".join(row) + "\n")
print("done", label, "steps", steps_seen, "drop", f"{loss_drop:.6f}")
PY
}

run_one nm_a 0 1 16 456 5 2 0.67
run_one nm_b 0 1 16 448 5 2 0.67
run_one moe_a 1 7 9 400 4 0 0.50
run_one moe_b 1 8 10 384 4 0 0.60

echo "DONE $SUMMARY"
