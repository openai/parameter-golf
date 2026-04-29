#!/bin/bash
# Phase Q — drop EMBED_CLIP_SIGMAS from 14 (Phase G's tightened) back to 20
# (PR #1797 default looser clip), based on the empirical observation that
# LOOSER clip → SMALLER artifact (counterintuitive but reproducibly true).
#
# Phase G (MLP=11.5, EMBED_CLIP=14) → 16.14 MB / 1.05969 mean
# Phase Q (MLP=11.5, EMBED_CLIP=20) → expected ~15.85-15.95 MB / ~1.0597
#
# This is the cheapest single-knob test left. If artifact fits AND BPB
# holds, we have a clean record-clearing 3-seed.
set -euo pipefail
set -x

BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
SEEDS=${SEEDS:-"42 314 1234"}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phq}
POD_ID=${RUNPOD_POD_ID:-}

HARD_DEADLINE_MIN=${HARD_DEADLINE_MIN:-100}
( sleep $((HARD_DEADLINE_MIN*60)) && [ -n "$POD_ID" ] && runpodctl stop pod "$POD_ID" ) &
KILL_PID=$!

cleanup() {
  kill $KILL_PID 2>/dev/null || true
  echo "[$(date)] === Phase Q done; auto-stop pod $POD_ID ==="
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

# --- PR #1797 V2 base + BOS fix + 9-hparam (Phase G) but EMBED_CLIP back to 20 ---
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
export MLP_CLIP_SIGMAS=11.5
# *** THE KEY DELTA: revert EMBED_CLIP_SIGMAS from 14 (9-hparam) back to 20 (PR #1797 default) ***
export EMBED_CLIP_SIGMAS=20.0
export WARMDOWN_FRAC=0.85
export BETA2=0.99
export TTT_BETA2=0.99
export TTT_WEIGHT_DECAY=0.5
export TTT_LORA_RANK=80
export SPARSE_ATTN_GATE_SCALE=0.5
export COMPRESSOR=brotli
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

  python3 - "$CSV" "$RUN_ID_PREFIX" <<'PY' || true
import sys, os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj=open(sys.argv[1], 'rb').read(),
    path_in_repo=f"results/{sys.argv[2]}_summary.csv",
    repo_id="FijaEE/parameter-golf-sp8192-caseops",
    repo_type="dataset", commit_message=f"{sys.argv[2]} progress",
)
PY

  if [ $SEED_NUM -eq 1 ] && [ -n "$ART" ]; then
    if [ "$ART" -gt 16000000 ]; then
      ABORT_REASON="seed-1 art $ART > 16M; EMBED_CLIP=20 not enough"
      echo "[$(date)] === ABORT: $ABORT_REASON ==="; break
    fi
    if [ -n "$TTT" ]; then
      cmp=$(python3 -c "print(int(float('$TTT') > 1.061))")
      [ "$cmp" = "1" ] && ABORT_REASON="seed-1 BPB $TTT > 1.061" && echo "[$(date)] === ABORT: $ABORT_REASON ===" && break
    fi
  fi
done

echo
echo "=== PHASE Q FINAL ==="
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
    print(f"vs PR #1797: delta={diff:+.6f}  Welch t={t:.2f}")
    rec_bar = 0.005 / math.log(2) / 3.7266
    print(f"record bar (0.005 nats ≈ {rec_bar:.5f} BPB):  {'CLEAR ✅' if -diff > rec_bar else 'MISS ❌'}")
print()
if art:
    valid = all(a <= 16_000_000 for a in art)
    print(f"All artifacts under 16 MB cap: {valid}  (max={max(art):,})")
PY
