#!/bin/bash
# Phase M — Last shot at record before deadline.
#
# Stack: PR #1797 V2 base + BOS fix (in code) + PR #1855 9-hparam stack +
#        COMPRESSOR=lzma (replaces brotli; ~5-15% smaller compressed output
#        on quantized weights, expected to bring artifact <16 MB after Phase G
#        was 16.14 MB with brotli + 9-hparam).
#
# CRITICAL: this script self-stops the pod after the 3-seed run finishes.
# That avoids the $180 idle waste from Phase K → Phase L gap.
set -euo pipefail
set -x

BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
SEEDS=${SEEDS:-"42 314 1234"}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phm}
POD_ID=${RUNPOD_POD_ID:-}  # set by RunPod pods automatically

# Hard wallclock kill switch — if anything hangs, stop pod after this many minutes.
# 3 seeds × ~22 min each + 10 min HF dl = ~76 min expected. 100 min is the safety net.
HARD_DEADLINE_MIN=${HARD_DEADLINE_MIN:-100}
( sleep $((HARD_DEADLINE_MIN*60)) && echo "[KILL] hard deadline hit, stopping pod" \
    && [ -n "$POD_ID" ] && runpodctl stop pod "$POD_ID" ) &
KILL_PID=$!

cleanup() {
  kill $KILL_PID 2>/dev/null || true
  echo "[$(date)] === Phase M complete; auto-stopping pod $POD_ID ==="
  if [ -n "$POD_ID" ]; then
    # Try runpodctl, fall back to GraphQL API
    runpodctl stop pod "$POD_ID" 2>&1 | head -3 || true
    if [ -n "${HF_TOKEN:-}" ]; then
      :  # placeholder
    fi
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
      curl -sS -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id } }\"}" 2>&1 | head -3
    fi
  fi
}
trap cleanup EXIT

export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_inductor
export TRITON_CACHE_DIR=/workspace/triton
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false NCCL_NET=Socket
mkdir -p /workspace/runs $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $HF_HOME

which pyminify >/dev/null 2>&1 || pip install --break-system-packages python-minifier brotli zstandard 2>&1 | tail -2

REPO=/workspace/parameter-golf
SUB=$REPO/records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix
if [ ! -d "$REPO/.git" ]; then
  rm -rf "$REPO"
  git clone --depth=1 --branch "$BRANCH" https://github.com/Fija/parameter-golf.git "$REPO"
else
  (cd "$REPO" && git fetch --depth=1 origin "$BRANCH" && git reset --hard "origin/$BRANCH")
fi
cd "$SUB"

DATA_DIR=/workspace/data/datasets/fineweb10B_sp8192_caseops
mkdir -p "$DATA_DIR"
TRAIN_SAMPLE=$DATA_DIR/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_000000.bin
if [ ! -f "$TRAIN_SAMPLE" ]; then
  python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="FijaEE/parameter-golf-sp8192-caseops",
    repo_type="dataset",
    local_dir="/workspace/data/datasets/fineweb10B_sp8192_caseops",
    token=os.environ["HF_TOKEN"], max_workers=16,
)
PY
fi

python3 test_ngram_legality.py

# --- PR #1797 V2 base + BOS fix (in code) + PR #1855 9-hparam stack ---
export DATA_PATH=$DATA_DIR/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
export TOKENIZER_PATH=$SUB/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192 CASEOPS_ENABLED=1
export MAX_WALLCLOCK_SECONDS=620
export FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1
export SMEAR_GATE_ENABLED=1 GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4
export LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export TTT_WARM_START_A=1 EMBED_BITS=7 MIN_LR=0.1 MATRIX_LR=0.026
export MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0
export GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16
export PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3
# --- PR #1855 9-hparam stack (delta vs PR #1797 in comments) ---
export MLP_CLIP_SIGMAS=11.5         # was 10.0
export EMBED_CLIP_SIGMAS=14.0       # was 20.0 (TIGHTER, helps artifact)
export WARMDOWN_FRAC=0.85           # was 0.75
export BETA2=0.99                   # was 0.95
export TTT_BETA2=0.99               # was 0.999
export TTT_WEIGHT_DECAY=0.5         # was 1.0
export TTT_LORA_RANK=80             # was 96
export SPARSE_ATTN_GATE_SCALE=0.5   # was 1.0
# --- THE KEY DELTA vs Phase G: switch compressor to lzma ---
export COMPRESSOR=lzma
# Mixers all OFF
export NGRAM_MIX_ENABLED=0 TEMP_SCALE_ENABLED=0 PPM_MIX_ENABLED=0

CSV=/workspace/runs/${RUN_ID_PREFIX}_summary.csv
echo "seed,pre_quant_bpb,post_quant_bpb,ttt_bpb,artifact_bytes,eval_time_s" > $CSV

ABORT_REASON=""
SEED_NUM=0
for SEED in $SEEDS; do
  SEED_NUM=$((SEED_NUM+1))
  RID=${RUN_ID_PREFIX}_s${SEED}
  RDIR=/workspace/runs/$RID
  rm -rf "$RDIR"; mkdir -p "$RDIR"
  export RUN_ID=$RID SEED=$SEED QUANTIZED_MODEL_PATH=$RDIR/model.bin
  echo "[$(date)] === SEED $SEED START (seed $SEED_NUM/3) ==="
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RDIR/train_gpt.out"
  PRE=$(grep -oE "diagnostic pre-quantization post-ema[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  POST=$(grep -oE "diagnostic quantized[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  TTT=$(grep -oE "quantized_ttt_phased[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  ART=$(grep -oE "Total submission size [^:]+: *[0-9]+ bytes" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9]+' | tail -1 || echo "")
  EV=$(grep -oE "total_eval_time:[0-9.]+s" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+' || echo "")
  echo "$SEED,$PRE,$POST,$TTT,$ART,$EV" >> $CSV
  echo "[$(date)] === SEED $SEED DONE: ttt=$TTT artifact=$ART ==="

  # Persist results to HF so we can read them even after pod auto-stop.
  python3 - "$CSV" "$RUN_ID_PREFIX" <<'PY' || true
import sys, os, json
from huggingface_hub import HfApi, upload_file
csv_path, prefix = sys.argv[1], sys.argv[2]
text = open(csv_path).read()
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=text.encode(),
    path_in_repo=f"results/{prefix}_summary.csv",
    repo_id="FijaEE/parameter-golf-sp8192-caseops",
    repo_type="dataset",
    commit_message=f"Phase {prefix} progress",
)
print(f"uploaded {csv_path} to HF results/{prefix}_summary.csv")
PY

  # ABORT GUARD: if seed 1's artifact is still over cap, no point doing more.
  if [ $SEED_NUM -eq 1 ] && [ -n "$ART" ]; then
    if [ "$ART" -gt 16000000 ]; then
      ABORT_REASON="seed-1 artifact $ART > 16M cap; lzma did not save enough"
      echo "[$(date)] === ABORT: $ABORT_REASON ==="
      break
    fi
    # Also abort if BPB is way worse than expected (>1.064)
    if [ -n "$TTT" ]; then
      cmp=$(python3 -c "print(int(float('$TTT') > 1.064))")
      if [ "$cmp" = "1" ]; then
        ABORT_REASON="seed-1 BPB $TTT > 1.064 (worse than baseline by 0.004)"
        echo "[$(date)] === ABORT: $ABORT_REASON ==="
        break
      fi
    fi
  fi
done

echo
echo "=== PHASE M FINAL SUMMARY ==="
cat $CSV
[ -n "$ABORT_REASON" ] && echo "ABORTED: $ABORT_REASON"

python3 - <<PY
import csv, math, statistics
rows = list(csv.DictReader(open("$CSV")))
ttt = [float(r["ttt_bpb"]) for r in rows if r.get("ttt_bpb")]
art = [int(r["artifact_bytes"]) for r in rows if r.get("artifact_bytes")]
if len(ttt) >= 2:
    m = statistics.mean(ttt); s = statistics.stdev(ttt) if len(ttt) > 1 else 0
    print(f"\nmean ttt_bpb = {m:.6f}   std = {s:.6f}   n={len(ttt)}")
    base_m, base_s = 1.06157, 0.00066
    diff = m - base_m
    if s > 0:
        se = math.sqrt(s**2/len(ttt) + base_s**2/3)
        t = diff / se
        print(f"vs PR #1797 (1.06157±0.00066): delta={diff:+.6f} BPB,  Welch t={t:.2f}")
    rec_bar_bpb = 0.005 / math.log(2) / 3.7266
    print(f"record bar (0.005 nats ≈ {rec_bar_bpb:.5f} BPB):  {'CLEAR ✅' if -diff > rec_bar_bpb else 'MISS ❌'}")
print()
if art:
    valid = all(a <= 16_000_000 for a in art)
    print(f"All artifacts under 16 MB cap: {valid}  (max={max(art):,})")
PY

# Cleanup trap will auto-stop the pod after this exits.
echo "[$(date)] phase M script done — pod auto-stop incoming"
