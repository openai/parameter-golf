#!/usr/bin/env bash
# ACTUAL 8xH100 COMPETITION RUN — from scratch, no checkpoints
# Usage: bash launch_8xh100_competition.sh
set -uo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"
mkdir -p logs artifacts

RUN_ID="comp_$(date +%Y%m%d_%H%M%S)"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "logs/${RUN_ID}_master.txt"; }

# Kill any existing training
pkill -9 -f "python3.*train_gpt.py" 2>/dev/null || true
sleep 2

log "═══════════════════════════════════════════════════════════════════════"
log "  8xH100 COMPETITION RUN"
log "  RUN_ID: $RUN_ID"
log "═══════════════════════════════════════════════════════════════════════"

# Best architecture from jj6 winner (no checkpoint loading)
# 11L attention, 2048 seq, int8+lzma, ~28M params
# TRAIN_BATCH_TOKENS scaled for 8xH100 (grad_accum=1 on 8 GPUs)
# Target: ~6000 steps in 600s wallclock on H100
env \
  PYTHONUNBUFFERED=1 \
  ARCH=attention \
  RESET_ON_BOS=1 \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=11 \
  MLP_ACTIVATION=leakyrelu2 \
  TRAIN_SEQ_LEN=2048 \
  WARMUP_STEPS=20 \
  SOFTCAP=10.0 \
  ROPE_DIMS=32 \
  QK_GAIN=1.0 \
  BWCE=1 \
  CGGR_RATIO=1.0 \
  TRIGRAM_HASH_SIZE=5120 \
  XSA_LAST_N=11 \
  SCALAR_LR=0.004 \
  TIED_EMBED_LR=0.005 \
  COMPILE_MODEL=1 \
  SEED=99 \
  VAL_MAX_TOKENS=4194304 \
  TRAIN_LOG_EVERY=100 \
  VAL_LOSS_EVERY=200 \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  MAX_WALLCLOCK_SECONDS=600 \
  ITERATIONS=5000 \
  WARMDOWN_ITERS=20 \
  SUBMIT_FMT=mixed_lzma \
  INT4_QUANT=1 \
  INT4_GROUP_SIZE=32 \
  INT4_STOCH_QUANT=1 \
  QAT_START_FRACTION=0.0 \
  DEPTH_RECUR=4 \
  EMA_DECAY=0.995 \
  RESIDUAL_SIGNS=1 \
  RESIDUAL_SIGNS_FILTER=ffn \
  RESIDUAL_SIGNS_FFN_LAYER=down \
  RESIDUAL_SIGNS_BLOCKS=0,2,3,5,6,7,8,9,10 \
  SELECTIVE_INT6_BLOCKS=0 \
  STOCH_EVAL_N=16 \
  STOCH_EVAL_EPS=0.5 \
  TTT_RANK=8 \
  TTT_STEPS=5 \
  TTT_LR=0.01 \
  TTT_CHUNK_SEQS=4 \
  TTT_MOMENTUM=0.9 \
  OUTLIER_FILTER=all \
  OUTLIER_HESSIAN=1 \
  OUTLIER_HESSIAN_BATCHES=8 \
  OUTLIER_TOPK_FRAC=0.0029 \
  OUTLIER_FRAC_TOP=0.00485 \
  OUTLIER_FRAC_BOTTOM=0.00095 \
  LOGIT_TEMP=1.02 \
  RUN_ID="$RUN_ID" \
  torchrun --standalone --nproc_per_node=8 "$REPO/train_gpt.py" 2>&1 | tee "logs/${RUN_ID}.txt"

# Extract results
LOG_F="logs/${RUN_ID}.txt"
log ""
log "═════════════════════  RESULT  ════════════════════"
RT=$(grep -E "final_(mixed_lzma|mixed_zstd|int8_zlib)_roundtrip_exact" "$LOG_F" 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "---")
SIZE=$(grep "Total submission size mixed+lzma:" "$LOG_F" 2>/dev/null | tail -1 | grep -oP '\d+' | head -1 || echo "---")
TRAIN_TIME=$(grep "train_time:" "$LOG_F" 2>/dev/null | tail -1 | grep -oP 'train_time:\K[0-9]+' || echo "---")
FINAL_VAL=$(grep "val_bpb:" "$LOG_F" 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "---")
OVER=$(grep -c "exceeds 16MB limit" "$LOG_F" 2>/dev/null || echo "0")

printf "  Roundtrip BPB:  %s\n" "$RT"
printf "  Pre-eval BPB:   %s\n" "$FINAL_VAL"
printf "  Train time:     %s ms\n" "$TRAIN_TIME"
printf "  Submission:    %s bytes\n" "$SIZE"
[[ "$OVER" -gt 0 ]] && echo "  STATUS:         *** OVER 16MB BUDGET ***" || echo "  STATUS:         OK"
log "══════════════════════════════════════════════════"