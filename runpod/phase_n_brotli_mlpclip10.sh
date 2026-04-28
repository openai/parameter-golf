#!/bin/bash
# Phase N — final record attempt:
#   - Same as Phase G (9-hparam stack giving mean 1.05969 BPB)
#   - BUT MLP_CLIP_SIGMAS reverted from 11.5 → 10.0 (PR #1797 default;
#     tighter MLP weight clip → smaller compressed quantized output)
#   - Brotli (better than lzma per Phase M finding)
#   - Auto-stop pod + abort guards from Phase M
#
# Hypothesis: Phase G was 16.14 MB (over by 144 KB) primarily because
# MLP_CLIP=11.5 widens weight magnitude. Reverting to 10.0 saves ~100-200
# KB at minimal BPB cost — Phase G's mean 1.05969 might shift to ~1.0598
# but should still clear 0.005 nat record bar (PR #1797 = 1.06157, bar at
# 1.05963).
set -euo pipefail
set -x

BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
SEEDS=${SEEDS:-"42 314 1234"}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phn}
POD_ID=${RUNPOD_POD_ID:-}

HARD_DEADLINE_MIN=${HARD_DEADLINE_MIN:-100}
( sleep $((HARD_DEADLINE_MIN*60)) && echo "[KILL] hard deadline" \
    && [ -n "$POD_ID" ] && runpodctl stop pod "$POD_ID" ) &
KILL_PID=$!

cleanup() {
  kill $KILL_PID 2>/dev/null || true
  echo "[$(date)] === Phase N done; auto-stopping pod $POD_ID ==="
  if [ -n "$POD_ID" ]; then
    runpodctl stop pod "$POD_ID" 2>&1 | head -3 || true
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

# --- PR #1797 V2 base + BOS fix (in code) + PR #1855 9-hparam stack except MLP_CLIP ---
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
# --- 8 of the 9-hparam stack (THE KEY DELTA: MLP_CLIP reverted to 10.0) ---
export MLP_CLIP_SIGMAS=10.0         # was 11.5 in Phase G/M; revert to PR #1797 default
export EMBED_CLIP_SIGMAS=14.0       # tighter than 20 (helps artifact)
export WARMDOWN_FRAC=0.85
export BETA2=0.99
export TTT_BETA2=0.99
export TTT_WEIGHT_DECAY=0.5
export TTT_LORA_RANK=80
export SPARSE_ATTN_GATE_SCALE=0.5
# Brotli compression (better than lzma on weight blobs per Phase M finding)
export COMPRESSOR=brotli
# Mixers OFF
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
  echo "[$(date)] === SEED $SEED ($SEED_NUM/3) ==="
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RDIR/train_gpt.out"
  PRE=$(grep -oE "diagnostic pre-quantization post-ema[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  POST=$(grep -oE "diagnostic quantized[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  TTT=$(grep -oE "quantized_ttt_phased[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  ART=$(grep -oE "Total submission size [^:]+: *[0-9]+ bytes" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9]+' | tail -1 || echo "")
  EV=$(grep -oE "total_eval_time:[0-9.]+s" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+' || echo "")
  echo "$SEED,$PRE,$POST,$TTT,$ART,$EV" >> $CSV
  echo "[$(date)] === SEED $SEED DONE: ttt=$TTT artifact=$ART ==="

  # Push results to HF
  python3 - "$CSV" "$RUN_ID_PREFIX" <<'PY' || true
import sys, os, json
from huggingface_hub import HfApi
csv_path, prefix = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=open(csv_path, 'rb').read(),
    path_in_repo=f"results/{prefix}_summary.csv",
    repo_id="FijaEE/parameter-golf-sp8192-caseops",
    repo_type="dataset",
    commit_message=f"{prefix} progress",
)
print(f"uploaded to HF results/{prefix}_summary.csv")
PY

  if [ $SEED_NUM -eq 1 ] && [ -n "$ART" ]; then
    if [ "$ART" -gt 16000000 ]; then
      ABORT_REASON="seed-1 artifact $ART > 16M; MLP_CLIP=10 not enough. Need EMBED_CLIP=12 next."
      echo "[$(date)] === ABORT: $ABORT_REASON ==="
      break
    fi
    if [ -n "$TTT" ]; then
      cmp=$(python3 -c "print(int(float('$TTT') > 1.064))")
      if [ "$cmp" = "1" ]; then
        ABORT_REASON="seed-1 BPB $TTT > 1.064"
        echo "[$(date)] === ABORT: $ABORT_REASON ==="
        break
      fi
    fi
  fi
done

echo
echo "=== PHASE N FINAL SUMMARY ==="
cat $CSV
[ -n "$ABORT_REASON" ] && echo "ABORTED: $ABORT_REASON"

python3 - <<PY
import csv, math, statistics
rows = list(csv.DictReader(open("$CSV")))
ttt = [float(r["ttt_bpb"]) for r in rows if r.get("ttt_bpb")]
art = [int(r["artifact_bytes"]) for r in rows if r.get("artifact_bytes")]
if len(ttt) >= 2:
    m = statistics.mean(ttt); s = statistics.stdev(ttt)
    print(f"\nmean ttt_bpb = {m:.6f}   std = {s:.6f}   n={len(ttt)}")
    base_m, base_s = 1.06157, 0.00066
    diff = m - base_m
    se = math.sqrt(s**2/len(ttt) + base_s**2/3)
    t = diff / se if se > 0 else 0
    print(f"vs PR #1797 (1.06157±0.00066): delta={diff:+.6f} BPB,  Welch t={t:.2f}")
    rec_bar_bpb = 0.005 / math.log(2) / 3.7266
    print(f"record bar (0.005 nats ≈ {rec_bar_bpb:.5f} BPB):  {'CLEAR ✅' if -diff > rec_bar_bpb else 'MISS ❌'}")
print()
if art:
    valid = all(a <= 16_000_000 for a in art)
    print(f"All artifacts under 16 MB cap: {valid}  (max={max(art):,})")
PY

echo "[$(date)] phase N done"
