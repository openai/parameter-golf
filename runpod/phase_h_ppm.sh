#!/bin/bash
# Phase H: PR #1797 base (no 9-hparam, V2 settings that fit 16MB) + BOS fix
# + Token-level PPM-D mixture (V3 of n-gram mixer).
#
# Reports BOTH the diagnostic eval_val_bpb AND the TTT phased number for
# completeness. The eval_val + PPM number is the headline (PPM is wired
# into eval_val, not eval_val_ttt_phased).
#
# Single seed by default — bumps to 3 once the 1-seed result is promising.
set -euo pipefail
set -x

BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
SEEDS=${SEEDS:-"42"}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phh}

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

# --- Common env (PR #1797 base, V2 = under-cap setup, NO 9-hparam stack) ---
export DATA_PATH=$DATA_DIR/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
export TOKENIZER_PATH=$SUB/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192 CASEOPS_ENABLED=1
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-620}
export FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1
export SMEAR_GATE_ENABLED=1 GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4
export LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export TTT_WARM_START_A=1 EMBED_BITS=7 MIN_LR=0.1 MATRIX_LR=0.026
export MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 EMBED_CLIP_SIGMAS=15.0
export GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16
export PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3

# Other mixers OFF; PPM-D ON
export NGRAM_MIX_ENABLED=0 TEMP_SCALE_ENABLED=0
export PPM_MIX_ENABLED=1
export PPM_MAX_ORDER=2
export PPM_LAMBDA_LO=0.05
export PPM_LAMBDA_HI=0.9
export PPM_CONF_THRESHOLD=0.9
export PPM_CHUNK_TOKENS=128

CSV=/workspace/runs/${RUN_ID_PREFIX}_summary.csv
echo "seed,pre_quant_bpb,post_quant_bpb,ttt_bpb,artifact_bytes,eval_time_s" > $CSV

for SEED in $SEEDS; do
  RID=${RUN_ID_PREFIX}_s${SEED}
  RDIR=/workspace/runs/$RID
  rm -rf "$RDIR"; mkdir -p "$RDIR"
  export RUN_ID=$RID SEED=$SEED QUANTIZED_MODEL_PATH=$RDIR/model.bin
  echo "[$(date)] === SEED $SEED START (PPM enabled) ==="
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RDIR/train_gpt.out"
  PRE=$(grep -oE "diagnostic pre-quantization post-ema[^\\n]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  POST=$(grep -oE "diagnostic quantized[^\\n]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  TTT=$(grep -oE "quantized_ttt_phased[^\\n]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  ART=$(grep -oE "Total submission size [^:]+: *[0-9]+ bytes" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9]+' | tail -1 || echo "")
  EVAL=$(grep -oE "total_eval_time:[0-9.]+s" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+' || echo "")
  echo "$SEED,$PRE,$POST,$TTT,$ART,$EVAL" >> $CSV
  echo "[$(date)] === SEED $SEED DONE: post_quant=$POST  ttt=$TTT  artifact=$ART ==="
done

echo
echo "=== PHASE H SUMMARY ==="
column -s, -t "$CSV"
