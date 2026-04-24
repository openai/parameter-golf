#!/bin/bash
# Phase B: Day-3 8xH100 sweep driver.
#
# Runs on an AP-IN-1 8xH100 Secure pod with NO Network Volume. Assumes
# bootstrap_no_volume.sh has already pulled the pre-tokenized SP8192 CaseOps
# shards from hf://datasets/FijaEE/parameter-golf-sp8192-caseops into
# /workspace/data/datasets/fineweb10B_sp8192_caseops.
#
# Phase B budget: 1 baseline train (~10 min) + 1 eval-only mixture-off pass +
# 9-point (alpha, beta) sweep (9 eval-only passes).  All eval-only passes
# reuse the one baseline quantized artifact.
#
# Pre-req env:
#   MAX_WALLCLOCK_SECONDS=620   (hard cap for training)
#   RUN_ID_PREFIX=phb1
# Optional knob defaults below.
#
# Emits per-run lines to /workspace/runs/phase_b_results.csv:
#   run_id,config,step,pre_ttt_bpb,post_ema_bpb,quant_bpb,ttt_bpb,elapsed_s

set -euo pipefail

export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_inductor
export TRITON_CACHE_DIR=/workspace/triton
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/runs

REPO=/workspace/parameter-golf
SUB=$REPO/records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix
cd "$SUB"

CSV=/workspace/runs/phase_b_results.csv
if [ ! -f "$CSV" ]; then
  echo "run_id,config,step,pre_ttt_bpb,post_ema_bpb,quant_bpb,ttt_bpb,elapsed_s" > "$CSV"
fi

# ---------- step 1: baseline training (produces quantized artifact reused by all sweep points) ----------
BASE_RUN=${RUN_ID_PREFIX:-phb1}_base_s42
BASE_DIR=/workspace/runs/$BASE_RUN
mkdir -p "$BASE_DIR"
export RUN_ID=$BASE_RUN
export SEED=${SEED:-42}
export DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
export TOKENIZER_PATH=$SUB/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192
export CASEOPS_ENABLED=1
export QUANTIZED_MODEL_PATH=$BASE_DIR/model.bin
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-620}

# Baseline: mixture OFF. Training uses PR #1797 defaults (SMEAR, LQER, phased TTT, etc.).
export NGRAM_MIX_ENABLED=0

if [ ! -f "$BASE_DIR/model.bin" ]; then
  echo "[$(date)] --- BASELINE TRAIN seed=$SEED ---" | tee -a "$BASE_DIR/driver.log"
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$BASE_DIR/train_gpt.out"
else
  echo "[$(date)] baseline artifact already exists at $BASE_DIR/model.bin — skipping train"
fi
BASE_BPB=$(grep -oE "quantized_ttt_phased[^\n]*val_bpb:[0-9.]+" "$BASE_DIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
echo "$BASE_RUN,baseline_mix_off,,,,,$BASE_BPB," >> "$CSV"
echo "[$(date)] baseline complete: $BASE_BPB"

# ---------- step 2: sweep (alpha, beta) — eval-only, reuse $BASE_DIR/model.bin ----------
# Reuse the trained artifact by pointing QUANTIZED_MODEL_PATH at it and
# setting PREQUANT_ONLY=0 but TRAIN_ONLY=0 + skipping training via a special path.
# Simplest: re-run train_gpt.py with a wallclock of 0 seconds so train loop
# exits early, then eval_val + eval_val_ttt_phased run on the existing artifact.

run_one() {
  local tag=$1 alpha=$2 beta=$3
  local rid=${RUN_ID_PREFIX:-phb1}_mix_${tag}
  local rdir=/workspace/runs/$rid
  mkdir -p "$rdir"
  export RUN_ID=$rid
  export NGRAM_MIX_ENABLED=1
  export NGRAM_MIX_ALPHA=$alpha
  export NGRAM_MIX_BETA=$beta
  export QUANTIZED_MODEL_PATH=$BASE_DIR/model.bin   # reuse baseline artifact
  export RESUME_FROM_QUANT=1   # signals script to skip training (see patch note below)
  export MAX_WALLCLOCK_SECONDS=1
  echo "[$(date)] --- SWEEP $tag alpha=$alpha beta=$beta ---" | tee -a "$rdir/driver.log"
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$rdir/train_gpt.out" || true
  MIX_BPB=$(grep -oE "quantized_ttt_phased[^\n]*val_bpb:[0-9.]+" "$rdir/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
  PRE_TTT=$(grep -oE "diagnostic quantized[^\n]*val_bpb:[0-9.]+" "$rdir/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$')
  echo "$rid,mix_${tag}_a${alpha}_b${beta},,,$PRE_TTT,,$MIX_BPB," >> "$CSV"
  echo "[$(date)] $tag done: pre_ttt=$PRE_TTT ttt=$MIX_BPB"
}

# 9-point grid (alpha, beta). Quick to iterate if each eval is ~90s on 8xH100.
run_one mix_off_sanity 1e9 0.0    # should reproduce baseline exactly
run_one a10_b010  1.0  -0.10
run_one a10_b025  1.0  -0.25
run_one a10_b040  1.0  -0.40
run_one a20_b010  2.0  -0.10
run_one a20_b025  2.0  -0.25
run_one a20_b040  2.0  -0.40
run_one a30_b025  3.0  -0.25
run_one a30_b040  3.0  -0.40

echo "[$(date)] Phase B sweep complete. Summary:"
column -s , -t "$CSV"
