#!/bin/bash
# Phase K — TTT params budget tradeoff.
#
# 1) Train SP8192 V2 baseline ONCE on 8xH100 (~10 min, saves model.bin to volume-less container disk).
# 2) Run 4 TTT_EVAL_ONLY=1 eval configs reusing the saved model.bin:
#      K0: grad_steps=1  prefix=2000  phases=3  ctx=2048   (V2 baseline reference)
#      K1: grad_steps=2  prefix=2000  phases=3  ctx=2048   (oracle bpb, expected over-budget)
#      K2: grad_steps=2  prefix=1500  phases=1  ctx=2048   (cut prefix + phases)
#      K3: grad_steps=2  prefix=2000  phases=3  ctx=1024   (cut ctx)
# 3) Print eval_time + ttt_bpb for each. The "best fit" is the lowest BPB
#    that's also <= 600s. That config is then used for Phase L.
set -euo pipefail
set -x

BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phk}

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

# --- Common env (PR #1797 V2 stack — same as Path G but no 9-hparam to keep artifact under cap) ---
export DATA_PATH=$DATA_DIR/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
export TOKENIZER_PATH=$SUB/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192 CASEOPS_ENABLED=1
export FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1
export SMEAR_GATE_ENABLED=1 GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4
export LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export TTT_WARM_START_A=1 EMBED_BITS=7 MIN_LR=0.1 MATRIX_LR=0.026
export MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 EMBED_CLIP_SIGMAS=15.0
export GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16
export NGRAM_MIX_ENABLED=0 TEMP_SCALE_ENABLED=0 PPM_MIX_ENABLED=0

# === Step 1: Train baseline (saves model.bin) ===
BASE_RUN=${RUN_ID_PREFIX}_base_s42
BASE_DIR=/workspace/runs/$BASE_RUN
rm -rf "$BASE_DIR"; mkdir -p "$BASE_DIR"
export RUN_ID=$BASE_RUN SEED=42
export QUANTIZED_MODEL_PATH=$BASE_DIR/model.bin
export MAX_WALLCLOCK_SECONDS=620
# K0 reference also uses these
export PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3
export TTT_EVAL_SEQ_LEN=2048
export TTT_GRAD_STEPS=1

echo "[$(date)] === STEP 1: TRAIN baseline (also produces K0 numbers) ==="
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$BASE_DIR/train_gpt.out"
K0_TTT=$(grep -oE "quantized_ttt_phased[^|]*val_bpb:[0-9.]+" "$BASE_DIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
K0_EVAL_T=$(grep -oE "total_eval_time:[0-9.]+s" "$BASE_DIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+')
K0_ART=$(grep -oE "Total submission size [^:]+: *[0-9]+ bytes" "$BASE_DIR/train_gpt.out" | tail -1 | grep -oE '[0-9]+' | tail -1)
echo "[$(date)] K0 baseline:  ttt=$K0_TTT  eval=${K0_EVAL_T}s  artifact=$K0_ART"

# === Step 2: Run 3 more eval-only configs (K1, K2, K3) reusing model.bin ===
# We use TTT_EVAL_ONLY=1 to skip training + serialize.

run_eval_cfg() {
  local tag=$1 grad_steps=$2 prefix=$3 phases=$4 ctx=$5
  local rid=${RUN_ID_PREFIX}_${tag}_s42
  local rdir=/workspace/runs/$rid
  rm -rf "$rdir"; mkdir -p "$rdir"
  export RUN_ID=$rid SEED=42
  # Reuse the trained artifact
  export QUANTIZED_MODEL_PATH=$BASE_DIR/model.bin
  export TTT_EVAL_ONLY=1
  export TTT_GRAD_STEPS=$grad_steps
  export PHASED_TTT_PREFIX_DOCS=$prefix
  export PHASED_TTT_NUM_PHASES=$phases
  export TTT_EVAL_SEQ_LEN=$ctx
  echo "[$(date)] === $tag: grad_steps=$grad_steps prefix=$prefix phases=$phases ctx=$ctx ==="
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$rdir/train_gpt.out"
  local TTT_BPB=$(grep -oE "quantized_ttt_phased[^|]*val_bpb:[0-9.]+" "$rdir/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  local EVAL_T=$(grep -oE "total_eval_time:[0-9.]+s" "$rdir/train_gpt.out" | tail -1 | grep -oE '[0-9.]+' || echo "")
  echo "[$(date)] $tag: ttt=$TTT_BPB  eval=${EVAL_T}s"
  echo "$tag,$grad_steps,$prefix,$phases,$ctx,$TTT_BPB,$EVAL_T" >> /workspace/runs/${RUN_ID_PREFIX}_summary.csv
}

echo "config,grad_steps,prefix,phases,ctx,ttt_bpb,eval_time_s" > /workspace/runs/${RUN_ID_PREFIX}_summary.csv
echo "K0,1,2000,3,2048,$K0_TTT,$K0_EVAL_T" >> /workspace/runs/${RUN_ID_PREFIX}_summary.csv

run_eval_cfg K1 2 2000 3 2048
run_eval_cfg K2 2 1500 1 2048
run_eval_cfg K3 2 2000 3 1024

echo
echo "=== PHASE K SUMMARY ==="
cat /workspace/runs/${RUN_ID_PREFIX}_summary.csv

# Print recommendation (lowest BPB among configs that fit 600s)
python3 - <<PY
import csv
rows = list(csv.DictReader(open("/workspace/runs/${RUN_ID_PREFIX}_summary.csv")))
fit = [r for r in rows if r['ttt_bpb'] and r['eval_time_s'] and float(r['eval_time_s']) <= 600.0]
print()
print("Configs fitting <=600s eval time:")
for r in fit:
    print(f"  {r['config']:4s}  bpb={r['ttt_bpb']}  eval={r['eval_time_s']}s")
if fit:
    best = min(fit, key=lambda r: float(r['ttt_bpb']))
    print(f"\\nBEST FIT: {best['config']}  bpb={best['ttt_bpb']}  eval={best['eval_time_s']}s")
    print(f"  -> grad_steps={best['grad_steps']} prefix={best['prefix']} phases={best['phases']} ctx={best['ctx']}")
else:
    print("WARNING: no config fits 600s — may need K4 with even more aggressive cuts")
PY
