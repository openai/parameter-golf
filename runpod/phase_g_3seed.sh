#!/bin/bash
# Path G: PR #1797 base + SmearGate BOS Fix (PR #1851, in-code) +
# PR #1855 9-hparam greedy stack (all env vars below).
#
# Runs the FULL train+eval pipeline 3 times for seeds 42, 314, 1234 and
# logs each result. After all 3, prints the mean ± std and a Welch t-test
# vs PR #1797's reported 1.06157 / std 0.00066.
#
# Expected mean BPB ≈ 1.058-1.060 (combining hw variance + 9-hparam +
# BOS fix); should clear the 0.005-nat (~0.00194 BPB) record bar over
# PR #1797 with p < 0.01.
set -euo pipefail
set -x

BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
SEEDS=${SEEDS:-"42 314 1234"}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-pg3s}

export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_inductor
export TRITON_CACHE_DIR=/workspace/triton
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false
export NCCL_NET=Socket
mkdir -p /workspace/runs $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $HF_HOME

# Ensure pyminify CLI present (python-minifier package, NOT pyminify)
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

# Data — pull from HF private dataset if not already present
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
    token=os.environ["HF_TOKEN"],
    max_workers=16,
)
PY
fi

# Legality smoke (must pass before we burn 3-seed compute)
python3 test_ngram_legality.py

# ---- Common env (all seeds) ----
export DATA_PATH=$DATA_DIR/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
export TOKENIZER_PATH=$SUB/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192 CASEOPS_ENABLED=1
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-620}

# PR #1797 base components
export FUSED_CE_ENABLED=1
export SPARSE_ATTN_GATE_ENABLED=1
export SMEAR_GATE_ENABLED=1 GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4
export LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export TTT_WARM_START_A=1
export EMBED_BITS=7
export MIN_LR=0.1
export MATRIX_LR=0.026
export MATRIX_CLIP_SIGMAS=12.85
export ATTN_CLIP_SIGMAS=13.0
export GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16

# Phased TTT
export PHASED_TTT_PREFIX_DOCS=2500
export PHASED_TTT_NUM_PHASES=3

# --- PR #1855 9-hparam greedy stack (delta vs PR #1797 in comment) ---
export MLP_CLIP_SIGMAS=11.5         # was 10.0
export EMBED_CLIP_SIGMAS=14.0       # was 20.0
export WARMDOWN_FRAC=0.85           # was 0.75
export BETA2=0.99                   # was 0.95
export TTT_BETA2=0.99               # was 0.999
export TTT_WEIGHT_DECAY=0.5         # was 1.0
export TTT_LORA_RANK=80             # was 96
export SPARSE_ATTN_GATE_SCALE=0.5   # was 1.0

# Mixers OFF — pure neural baseline plus BOS fix + 9-hparam stack
export NGRAM_MIX_ENABLED=0
export TEMP_SCALE_ENABLED=0

CSV=/workspace/runs/${RUN_ID_PREFIX}_summary.csv
echo "seed,steps,pre_quant_bpb,post_quant_bpb,ttt_bpb,artifact_bytes,train_time_s,eval_time_s" > $CSV

for SEED in $SEEDS; do
  RID=${RUN_ID_PREFIX}_s${SEED}
  RDIR=/workspace/runs/$RID
  rm -rf "$RDIR"; mkdir -p "$RDIR"
  export RUN_ID=$RID SEED=$SEED
  export QUANTIZED_MODEL_PATH=$RDIR/model.bin
  echo "[$(date)] === SEED $SEED START ==="
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RDIR/train_gpt.out"
  # extract metrics
  PREQUANT=$(grep -oE "diagnostic pre-quantization post-ema[^\n]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
  POSTQUANT=$(grep -oE "diagnostic quantized[^\n]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
  TTTBPB=$(grep -oE "quantized_ttt_phased[^\n]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
  STEPS=$(grep -oE "step *[0-9]+/[0-9]+ train_loss" "$RDIR/train_gpt.out" | tail -1 | grep -oE "step *[0-9]+" | grep -oE "[0-9]+")
  ARTIFACT=$(grep -oE "Total submission size[^:]+: *[0-9]+ bytes" "$RDIR/train_gpt.out" | tail -1 | grep -oE "[0-9]+" | tail -1)
  TRAIN_T=$(grep -oE "stopping_early.*time_s:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
  [ -z "$TRAIN_T" ] && TRAIN_T=$(grep -oE "train_time: *[0-9.]+m" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+')
  EVAL_T=$(grep -oE "total_eval_time:[0-9.]+s" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+')
  echo "$SEED,$STEPS,$PREQUANT,$POSTQUANT,$TTTBPB,$ARTIFACT,$TRAIN_T,$EVAL_T" >> $CSV
  echo "[$(date)] === SEED $SEED DONE: ttt=$TTTBPB artifact=$ARTIFACT bytes ==="
done

# ---- Final summary ----
python3 - <<PY
import csv, math, statistics
rows = list(csv.DictReader(open("$CSV")))
ttt = [float(r["ttt_bpb"]) for r in rows if r.get("ttt_bpb")]
art = [int(r["artifact_bytes"]) for r in rows if r.get("artifact_bytes")]
print()
print("======= PATH G 3-SEED SUMMARY =======")
for r in rows:
    print(f"  seed={r['seed']:>4}  ttt_bpb={r['ttt_bpb']}  artifact={r['artifact_bytes']}  steps={r['steps']}  eval={r['eval_time_s']}s")
if len(ttt) >= 2:
    m = statistics.mean(ttt)
    s = statistics.stdev(ttt)
    print(f"\nmean ttt_bpb = {m:.6f}   std = {s:.6f}")
    # vs PR #1797: 1.06157 mean, 0.00066 std
    base_m, base_s = 1.06157, 0.00066
    diff = m - base_m
    se = math.sqrt(s**2/len(ttt) + base_s**2/3)
    t = diff / se if se > 0 else float("nan")
    print(f"vs PR #1797 (1.06157±0.00066): delta={diff:+.6f} BPB,  Welch t={t:.2f}")
    rec_bar_bpb = 0.005 / math.log(2) / 3.7266   # 0.005 nats SP8192
    print(f"record bar (0.005 nats ≈ {rec_bar_bpb:.5f} BPB):  {'CLEAR ✅' if -diff > rec_bar_bpb else 'MISS ❌'}")
print()
print(f"All artifacts under 16 MB cap: {all(a <= 16_000_000 for a in art)}")
PY
