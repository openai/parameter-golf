#!/bin/bash
# Phase B V2 — temperature-scaling knob.  Runs ON the 8xH100 pod.
#
# Differences vs V1:
#   1. Sets the FULL PR #1797 Run-Command env vars so artifact comes back
#      under 16 MB (V1 was 16.93 MB because EMBED_BITS=8 default vs PR #1797
#      uses 7, plus SMEAR/SPARSE/MIN_LR were unset).
#   2. New TemperatureScaler knob is the V2 sweep target.

set -euo pipefail
set -x

SEED=${SEED:-42}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phb2}
BRANCH=${BRANCH:-submission/pr1797-ngram-mix}

export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_inductor
export TRITON_CACHE_DIR=/workspace/triton
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/runs $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $HF_HOME

# python-minifier provides the 'pyminify' CLI (NOT the placeholder pyminify).
which pyminify >/dev/null 2>&1 || pip install --quiet --break-system-packages python-minifier brotli zstandard 2>&1 | tail -2 || true

REPO=/workspace/parameter-golf
SUB=$REPO/records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix
if [ ! -d "$REPO/.git" ]; then
  git clone --depth=1 --branch "$BRANCH" https://github.com/Fija/parameter-golf.git "$REPO"
else
  (cd "$REPO" && git fetch --depth=1 origin "$BRANCH" && git checkout "$BRANCH" && git reset --hard "origin/$BRANCH")
fi

# data already on volume (or pulled by previous run); pull if missing.
DATA_DIR=/workspace/data/datasets/fineweb10B_sp8192_caseops
if ! ls "$DATA_DIR/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_000000.bin" >/dev/null 2>&1; then
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

cd "$SUB"
python3 test_ngram_legality.py

# ----- baseline (PR #1797 full env, NO mixers) -----
BASE_RUN=${RUN_ID_PREFIX}_base_s${SEED}
BASE_DIR=/workspace/runs/$BASE_RUN
rm -rf "$BASE_DIR"; mkdir -p "$BASE_DIR"

# Common env (matches PR #1797 README "Run command (3-seed reproduction)")
export RUN_ID=$BASE_RUN SEED=$SEED
export DATA_PATH=$DATA_DIR/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
export TOKENIZER_PATH=$SUB/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192 CASEOPS_ENABLED=1
export QUANTIZED_MODEL_PATH=$BASE_DIR/model.bin
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-620}

# === PR #1797 full env block ===
export NCCL_NET=Socket
export PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3
export MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0
export MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0
export MATRIX_LR=0.026
export MIN_LR=0.1
export FUSED_CE_ENABLED=1
export SPARSE_ATTN_GATE_ENABLED=1
export SMEAR_GATE_ENABLED=1 GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4
export LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export TTT_WARM_START_A=1
export GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16
# === both knobs OFF for baseline ===
export NGRAM_MIX_ENABLED=0
export TEMP_SCALE_ENABLED=0

torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$BASE_DIR/train_gpt.out"

BASE_DIAG=$(grep -oE "diagnostic quantized[^\n]*val_bpb:[0-9.]+" "$BASE_DIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
BASE_TTT=$(grep -oE "quantized_ttt_phased[^\n]*val_bpb:[0-9.]+" "$BASE_DIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
BASE_BYTES=$(grep -oE "Total submission size quantized\+brotli: [0-9]+ bytes" "$BASE_DIR/train_gpt.out" | tail -1 | grep -oE '[0-9]+ bytes' | awk '{print $1}')
echo "[phb v2] baseline: diagnostic=$BASE_DIAG  ttt=$BASE_TTT  artifact=$BASE_BYTES bytes"

# ----- temperature sweep on the baseline artifact -----
SWEEP_OUT=/workspace/runs/${RUN_ID_PREFIX}_sweep.jsonl
rm -f "$SWEEP_OUT"
export SWEEP_OUTPUT=$SWEEP_OUT

torchrun --standalone --nproc_per_node=8 sweep_eval_val.py 2>&1 | tee /workspace/runs/${RUN_ID_PREFIX}_sweep.log

# ----- summary -----
python3 - <<PY
import json
rows = [json.loads(l) for l in open("$SWEEP_OUT").read().splitlines() if l.strip()]
rows.sort(key=lambda r: r.get("val_bpb", 1e9))
base = $BASE_DIAG
print(f"\n=== Phase B V2 sweep summary  (baseline diag eval_val = {base:.5f}) ===")
print(f"{'config':16s} {'kind':6s} {'val_bpb':>9s} {'delta':>9s} {'elapsed_s':>9s}")
for r in rows:
    d = r["val_bpb"] - base
    print(f"{r['config']:16s} {r.get('kind',''):6s} {r['val_bpb']:9.5f}  {d:+.5f} {r.get('elapsed_s',0):9.2f}")
print()
print(f"baseline TTT phased val_bpb = $BASE_TTT")
print(f"baseline artifact bytes     = $BASE_BYTES (cap = 16,000,000)")
PY

echo "[phb v2] DONE — see $SWEEP_OUT"
