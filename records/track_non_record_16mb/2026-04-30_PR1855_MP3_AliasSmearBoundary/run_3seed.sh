#!/bin/bash
# 3-seed runner for PR#1855 stack + MP3 marker-pair fusion + alias smear boundary.
# Built on the PR#1855 train_gpt.py (LQER + SparseAttnGate + BOS-fixed SmearGate +
# Polar-Express Muon + phased TTT eval + 9-hparam stack), with MP3 vocab surgery
# applied on top. SmearGate is fully disabled at positions immediately following
# an alias token (ALIAS_PREV_SMEAR_SCALE=0.0); regular positions are unchanged.
#
# Usage:
#   bash run_3seed.sh
# Env overrides:
#   SEEDS="42 0 1234"   # default (matches PR #1855 author convention)
#   DATA_PATH=...        # default ./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3
#   TOKENIZER_PATH=...   # default ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
set -o pipefail

SEEDS="${SEEDS:-42 0 1234}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"

if [ ! -d "$DATA_PATH" ]; then
  echo "ERROR: dataset dir not found: $DATA_PATH"
  echo "Run download_docs.py, prepare_caseops_data.py, prepare_marker_pair_v3.py first."
  exit 1
fi
if [ ! -f "$TOKENIZER_PATH" ]; then
  echo "ERROR: tokenizer model not found: $TOKENIZER_PATH"
  exit 1
fi

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === 5-seed run START ==="
echo "  SEEDS:          $SEEDS"
echo "  DATA_PATH:      $DATA_PATH"
echo "  TOKENIZER_PATH: $TOKENIZER_PATH"

for SEED in $SEEDS; do
  LOG="train_seed${SEED}.log"
  echo ""
  echo "[$(ts)] >>> seed=$SEED -> $LOG >>>"
  env SEED="$SEED" \
    CASEOPS_ENABLED=1 \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
    PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
    EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
    MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
    GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
    GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
    TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
    SPARSE_ATTN_GATE_SCALE=0.5 \
    GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
    GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
    SMEAR_GATE_ENABLED=1 \
    LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
    FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
    MARKER_PAIR_MODE=1 \
    MARKER_PAIR_W_SPACE=0.4 \
    MARKER_PAIR_W_TITLE=0.6 \
    ALIAS_PREV_SMEAR_SCALE=0.0 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG"
  rc="${PIPESTATUS[0]}"
  echo "[$(ts)] <<< seed=$SEED done (rc=$rc) <<<"
done

echo ""
echo "[$(ts)] === 5-seed run DONE ==="
echo ""
echo "=== Summary ==="
printf "%-8s %-14s %-14s %-12s %s\n" "seed" "ttt_phased_bpb" "val_loss" "size" "step_avg"
echo "------------------------------------------------------------------------"
for SEED in $SEEDS; do
  LOG="train_seed${SEED}.log"
  sw=$(grep -E "quantized_ttt_phased" "$LOG" 2>/dev/null | grep -oP "val_bpb:\K[0-9.]+" | tail -1)
  vl=$(grep -E "quantized_ttt_phased" "$LOG" 2>/dev/null | grep -oP "val_loss:\K[0-9.]+" | tail -1)
  sz=$(grep "Total submission size quantized+pergroup" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  st=$(grep -oP "step_avg: [0-9.]+" "$LOG" 2>/dev/null | tail -1)
  printf "%-8s %-14s %-14s %-12s %s\n" "$SEED" "${sw:-?}" "${vl:-?}" "${sz:-?}" "${st:-?}"
done
