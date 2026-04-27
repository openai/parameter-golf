#!/bin/bash
# Runs ON the 8xH100 AP-IN-1 pod. Orchestrates: bootstrap -> baseline train ->
# eval_val sweep -> TTT sweep on top-k configs -> print summary.
#
# Precondition:
#   - HF_TOKEN env is set (passed via runpodctl --env when the pod was created)
#   - This script lives at /workspace/runpod/phase_b_onpod.sh (rsync'd in)
#   - The bootstrap is already committed to the fork so we can git clone.
#
# Usage on pod:
#   /workspace/runpod/phase_b_onpod.sh  [SEED=42]  [RUN_ID_PREFIX=phb1]
set -euo pipefail
set -x

SEED=${SEED:-42}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phb1}
BRANCH=${BRANCH:-submission/pr1797-ngram-mix}

export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_inductor
export TRITON_CACHE_DIR=/workspace/triton
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=0
mkdir -p /workspace/runs $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $HF_HOME

# ---------- 1. bootstrap ----------
pip install --quiet --break-system-packages brotli zstandard python-minifier 2>&1 | tail -2 || true
# sanity-check pyminify CLI is installed (provided by python-minifier, NOT pyminify)
which pyminify || { echo "FATAL: pyminify CLI missing — python-minifier install failed"; exit 1; }

REPO=/workspace/parameter-golf
if [ ! -d "$REPO/.git" ]; then
  git clone --depth=1 --branch "$BRANCH" https://github.com/Fija/parameter-golf.git "$REPO"
else
  (cd "$REPO" && git fetch --depth=1 origin "$BRANCH" && git checkout "$BRANCH" && git reset --hard "origin/$BRANCH")
fi

SUB=$REPO/records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix

# ---------- 2. data pull from HF private dataset ----------
DATA_DIR=/workspace/data/datasets/fineweb10B_sp8192_caseops
mkdir -p "$DATA_DIR"
if ls "$DATA_DIR/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_000000.bin" >/dev/null 2>&1; then
  echo "[phb] data already present — skipping HF pull"
else
  python3 - <<'PY'
import os, sys
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="FijaEE/parameter-golf-sp8192-caseops",
    repo_type="dataset",
    local_dir="/workspace/data/datasets/fineweb10B_sp8192_caseops",
    token=os.environ["HF_TOKEN"],
    max_workers=16,
)
print("download complete:", p)
PY
fi

# ---------- 3. run legality self-test (confirms env parity) ----------
cd "$SUB"
python3 test_ngram_legality.py

# ---------- 4. baseline training (mixture OFF) ----------
BASE_RUN=${RUN_ID_PREFIX}_base_s${SEED}
BASE_DIR=/workspace/runs/$BASE_RUN
mkdir -p "$BASE_DIR"
export RUN_ID=$BASE_RUN
export SEED=$SEED
export DATA_PATH=$DATA_DIR/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
export TOKENIZER_PATH=$SUB/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192
export CASEOPS_ENABLED=1
export QUANTIZED_MODEL_PATH=$BASE_DIR/model.bin
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-620}
export NGRAM_MIX_ENABLED=0

if [ ! -f "$QUANTIZED_MODEL_PATH" ]; then
  echo "[phb] baseline train (no mixer)..."
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$BASE_DIR/train_gpt.out"
else
  echo "[phb] baseline model.bin already exists — skipping train"
fi
BASE_TTT_BPB=$(grep -oE "quantized_ttt_phased[^\n]*val_bpb:[0-9.]+" "$BASE_DIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
BASE_DIAG_BPB=$(grep -oE "diagnostic quantized[^\n]*val_bpb:[0-9.]+" "$BASE_DIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
echo "[phb] baseline: diagnostic=$BASE_DIAG_BPB  ttt=$BASE_TTT_BPB"

# ---------- 5. fast eval_val sweep (mixer on) reusing baseline artifact ----------
SWEEP_OUT=/workspace/runs/${RUN_ID_PREFIX}_sweep.jsonl
rm -f "$SWEEP_OUT"
export SWEEP_OUTPUT=$SWEEP_OUT

torchrun --standalone --nproc_per_node=8 sweep_eval_val.py 2>&1 | tee /workspace/runs/${RUN_ID_PREFIX}_sweep.log

# ---------- 6. summarize ----------
python3 - <<PY
import json, sys
path = "$SWEEP_OUT"
base_diag = $BASE_DIAG_BPB
rows = [json.loads(l) for l in open(path).read().splitlines() if l.strip()]
rows.sort(key=lambda r: r.get("val_bpb", 1e9))
print(f"\n=== sweep summary (baseline diagnostic eval_val = {base_diag:.5f}) ===")
print(f"{'config':28s} {'val_bpb':>9s} {'delta':>8s} {'elapsed_s':>9s}")
for r in rows:
    d = r["val_bpb"] - base_diag
    print(f"{r['config']:28s} {r['val_bpb']:9.5f} {d:+.5f} {r.get('elapsed_s','?'):>9}")
PY

echo "[phb] DONE — results in $SWEEP_OUT and /workspace/runs/${RUN_ID_PREFIX}_sweep.log"
