#!/bin/bash
# Phase L — Combo decisive 3-seed run.
#
# Stack: PR #1797 V2 base (BOS fix already in code) + SP10240 CaseOps
#        + best TTT params from Phase K + LoRA rank 96.
#
# Best TTT params come in via env (set by caller after Phase K). Defaults
# below are the safest baseline (PR #1797 V2 settings) — override if K shows
# something better.
set -euo pipefail
set -x

BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
SEEDS=${SEEDS:-"42 314 1234"}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phl}

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

# --- Pull SP10240 dataset ---
DATA_DIR=/workspace/data/datasets/fineweb10B_sp10240_caseops
mkdir -p "$DATA_DIR"
SAMPLE=$DATA_DIR/datasets/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved/fineweb_train_000000.bin
if [ ! -f "$SAMPLE" ]; then
  python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="FijaEE/parameter-golf-sp10240-caseops",
    repo_type="dataset",
    local_dir="/workspace/data/datasets/fineweb10B_sp10240_caseops",
    token=os.environ["HF_TOKEN"], max_workers=16,
)
PY
fi

python3 test_ngram_legality.py

# --- Common env: PR #1797 V2 stack + SP10240 ---
export DATA_PATH=$DATA_DIR/datasets/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved/
export TOKENIZER_PATH=$SUB/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=10240 CASEOPS_ENABLED=1
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-620}
export FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1
export SMEAR_GATE_ENABLED=1 GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4
export LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export TTT_WARM_START_A=1 EMBED_BITS=7 MIN_LR=0.1 MATRIX_LR=0.026
export MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 EMBED_CLIP_SIGMAS=15.0
export GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16

# --- TTT params: defaults are PR #1797 V2 ---
# Override via outer env after Phase K identifies the best fit.
export TTT_GRAD_STEPS=${TTT_GRAD_STEPS:-1}
export PHASED_TTT_PREFIX_DOCS=${PHASED_TTT_PREFIX_DOCS:-2000}
export PHASED_TTT_NUM_PHASES=${PHASED_TTT_NUM_PHASES:-3}
export TTT_EVAL_SEQ_LEN=${TTT_EVAL_SEQ_LEN:-2048}
export TTT_LORA_RANK=${TTT_LORA_RANK:-96}

export NGRAM_MIX_ENABLED=0 TEMP_SCALE_ENABLED=0 PPM_MIX_ENABLED=0

CSV=/workspace/runs/${RUN_ID_PREFIX}_summary.csv
echo "seed,pre_quant_bpb,post_quant_bpb,ttt_bpb,artifact_bytes,eval_time_s" > $CSV

for SEED in $SEEDS; do
  RID=${RUN_ID_PREFIX}_s${SEED}
  RDIR=/workspace/runs/$RID
  rm -rf "$RDIR"; mkdir -p "$RDIR"
  export RUN_ID=$RID SEED=$SEED QUANTIZED_MODEL_PATH=$RDIR/model.bin
  echo "[$(date)] === SEED $SEED START ==="
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RDIR/train_gpt.out"
  PRE=$(grep -oE "diagnostic pre-quantization post-ema[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  POST=$(grep -oE "diagnostic quantized[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  TTT=$(grep -oE "quantized_ttt_phased[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  ART=$(grep -oE "Total submission size [^:]+: *[0-9]+ bytes" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9]+' | tail -1 || echo "")
  EV=$(grep -oE "total_eval_time:[0-9.]+s" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+' || echo "")
  echo "$SEED,$PRE,$POST,$TTT,$ART,$EV" >> $CSV
  echo "[$(date)] === SEED $SEED DONE: post_quant=$POST ttt=$TTT artifact=$ART ==="
done

echo
echo "=== PHASE L SUMMARY (SP10240 + ${TTT_GRAD_STEPS}-step TTT) ==="
cat $CSV

python3 - <<PY
import csv, math, statistics
rows = list(csv.DictReader(open("$CSV")))
ttt = [float(r["ttt_bpb"]) for r in rows if r.get("ttt_bpb")]
art = [int(r["artifact_bytes"]) for r in rows if r.get("artifact_bytes")]
if len(ttt) >= 2:
    m = statistics.mean(ttt); s = statistics.stdev(ttt)
    print(f"\\nmean ttt_bpb = {m:.6f}   std = {s:.6f}")
    base_m, base_s = 1.06157, 0.00066
    diff = m - base_m
    se = math.sqrt(s**2/len(ttt) + base_s**2/3)
    t = diff / se if se > 0 else float("nan")
    print(f"vs PR #1797 (1.06157±0.00066): delta={diff:+.6f} BPB,  Welch t={t:.2f}")
    rec_bar_bpb = 0.005 / math.log(2) / 3.7266
    print(f"record bar (0.005 nats ≈ {rec_bar_bpb:.5f} BPB):  {'CLEAR ✅' if -diff > rec_bar_bpb else 'MISS ❌'}")
print()
print(f"All artifacts under 16 MB cap: {all(a <= 16_000_000 for a in art)}  (max={max(art) if art else '?'})")
PY
