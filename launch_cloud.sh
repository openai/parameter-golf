#!/usr/bin/env bash
# =================================================================
# Parameter Golf — Cloud Launch Script for RunPod 8×H100 SXM
# SOTA config: 1.1194 BPB (LeakyReLU² + Legal TTT + Parallel Muon)
# =================================================================
# Usage (on RunPod pod):
#   bash launch_cloud.sh           # Full 10-min competition run
#   bash launch_cloud.sh smoke     # Quick 3-iter sanity check
#   bash launch_cloud.sh medium    # 200-iter timing test, no TTT
#   bash launch_cloud.sh nottt     # Full training, skip TTT phase
#
# Pre-requisites: RunPod official Parameter Golf template, or any
#   H100-SXM pod with PyTorch + flash-attn-3 pre-installed:
#   https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
#
# Target performance: ~1.1194 BPB, ~15.9 MB artifact, 8×H100 SXM
#   Training: ~600s | Eval (int6 + sliding + TTT): ~530s extra
# =================================================================

set -euo pipefail

MODE="${1:-full}"
REPO_DIR="/workspace/parameter-golf"
FORK_URL="https://github.com/Omrigotlieb/parameter-golf.git"
# Use the SOTA record script which has all env-var-driven features:
# LN_SCALE, VE, BIGRAM_VOCAB_SIZE, XSA_LAST_N, SWA, TTT, Parallel Muon.
TRAIN_SCRIPT="records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py"
DATA_VARIANT="sp1024"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"

echo "===== Parameter Golf — Cloud Launch (8×H100 SXM) ====="
echo "Mode  : $MODE"
echo "Script: $TRAIN_SCRIPT"
echo "======================================================="

# ---------------------------------------------------------------
# 1. Clone / update repo
# ---------------------------------------------------------------
echo ""
echo "[1/5] Setting up repo..."
if [ -d "$REPO_DIR/.git" ]; then
    echo "  Repo exists — pulling latest main..."
    cd "$REPO_DIR"
    git fetch origin
    git reset --hard origin/main
else
    echo "  Cloning fresh..."
    cd /workspace
    git clone "$FORK_URL" parameter-golf
    cd "$REPO_DIR"
fi
echo "  Commit: $(git rev-parse --short HEAD)"

# ---------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------
echo ""
echo "[2/5] Installing dependencies..."
# zstandard: required for zstd-22 compression of the artifact
pip install -q zstandard 2>/dev/null || pip install zstandard
python3 -c "import zstandard; print('  zstandard OK')"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}')"
python3 -c "
try:
    from flash_attn_interface import flash_attn_func
    print('  FlashAttention-3 OK')
except ImportError:
    print('  WARNING: flash_attn_interface not found — will use PyTorch SDPA fallback')
"

# ---------------------------------------------------------------
# 3. Download data
# ---------------------------------------------------------------
echo ""
echo "[3/5] Checking data..."
DATA_PATH="./data/datasets/fineweb10B_${DATA_VARIANT}"
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

SHARD_COUNT=$(ls "$DATA_PATH"/fineweb_train_*.bin 2>/dev/null | wc -l || echo 0)
VAL_COUNT=$(ls  "$DATA_PATH"/fineweb_val_*.bin   2>/dev/null | wc -l || echo 0)
if [ "$SHARD_COUNT" -lt 10 ] || [ "$VAL_COUNT" -lt 1 ]; then
    echo "  Downloading FineWeb (shards=$TRAIN_SHARDS)..."
    python3 data/cached_challenge_fineweb.py \
        --variant "$DATA_VARIANT" \
        --train-shards "$TRAIN_SHARDS"
else
    echo "  Data already present: $SHARD_COUNT train shards, $VAL_COUNT val shards."
fi

# ---------------------------------------------------------------
# 4. Smoke test (skip in full/nottt modes)
# ---------------------------------------------------------------
if [ "$MODE" = "full" ] || [ "$MODE" = "nottt" ]; then
    echo ""
    echo "[4/5] Quick smoke test (2 GPUs, 3 iterations)..."
    NCCL_IB_DISABLE=1 \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE=1024 \
    ITERATIONS=3 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 \
    WARMUP_STEPS=1 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=1024 \
    TTT_ENABLED=0 SWA_ENABLED=0 VE_ENABLED=0 \
    torchrun --standalone --nproc_per_node=2 "$TRAIN_SCRIPT" 2>&1 | \
        grep -E "step:|final|int6|Error|Traceback" | head -20
    echo "  Smoke test passed."
fi

# ---------------------------------------------------------------
# 5. Main training run
# ---------------------------------------------------------------
echo ""
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ------------------------------------------------------------------
# BEST-CONFIG ENV VARS (sourced from 2026-03-23_LeakyReLU_LegalTTT)
# ------------------------------------------------------------------
# Architecture
# 11L × 512d, 8 heads, 4 KV heads (GQA), 3× MLP, seq_len=2048
# LeakyReLU(0.5)² activation (built into the record script's MLP)
# Partial RoPE: only 16 of 64 head dims rotated (better capacity use)
# LN Scale: 1/√(layer+1) damping of residual-stream variance
# XSA on last 4 layers: Exclusive Self Attention subtracts v-aligned component
# BigramHash 1536 buckets → compact bigram position signal
# SmearGate: trainable token mixing at the embedding layer
# ValueEmbedding (dim=128) on layers 9 and 10

# Optimizer
# Muon (matrix params): lr=0.025, momentum warm-up 0.92→0.99 over 1500 steps
#   + weight_decay=0.04, Parallel Muon (batched NS5 via torch.bmm)
# AdamW (scalars + embeddings): lr=0.025 / 0.035, WD=0.04

# Weight averaging
# EMA decay=0.997 every step → primary artifact weights
# Tight SWA every 50 steps during warmdown → stacks with EMA

# Quantization
# GPTQ-lite int6 with 5-percentile clip search per row (MLP + attn)
# Int8 per-row for embeddings
# Control tensors (scales, gains) stored in fp32
# Late QAT: STE int6 fake-quant when LR scale < 0.15
# Final compression: zstd level 22

# TTT (test-time training) — Legal score-first protocol
# Split val into 32K-token chunks; score each under inference_mode(),
# then adapt on already-scored tokens with SGD(lr=0.002, momentum=0.9)
# for 3 epochs; no weight mutation possible during scoring step.
# Contributes ~-0.0025 BPB, takes ~410s of the eval budget.

# Timing budget (8×H100 SXM):
#   Training: 600s (≤10 min)
#   Standard eval (int6 roundtrip + sliding window s64): ~120s
#   Legal TTT (score-first): ~410s
#   Total eval: ~530s  →  Grand total: ~1130s (~19 min, within rules)
# ------------------------------------------------------------------

case "$MODE" in

  smoke)
    echo "[4/5] Smoke mode: 200 iters, val every 100, no TTT"
    NCCL_IB_DISABLE=1 \
    RUN_ID="cloud_smoke_${RUN_TIMESTAMP}" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE=1024 \
    \
    NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
    BIGRAM_VOCAB_SIZE=1536 BIGRAM_DIM=128 \
    XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
    VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10" \
    TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0 \
    MTP_NUM_HEADS=3 MTP_LOSS_WEIGHT=0.15 \
    Z_LOSS_WEIGHT=0.0001 QK_GAIN_INIT=1.5 \
    \
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_BACKEND_STEPS=5 \
    MUON_WD=0.04 ADAM_WD=0.04 \
    BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 \
    \
    ITERATIONS=200 WARMDOWN_ITERS=50 WARMUP_STEPS=5 \
    TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 \
    MAX_WALLCLOCK_SECONDS=0 \
    VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=20 \
    EVAL_STRIDE=64 \
    \
    EMA_DECAY=0.997 \
    SWA_ENABLED=1 SWA_EVERY=50 \
    LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    TTT_ENABLED=0 \
    \
    SEED="${SEED:-1337}" \
    torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"
    ;;

  medium)
    echo "[4/5] Medium mode: 500 iters (~40s on 8xH100), no TTT — timing probe"
    NCCL_IB_DISABLE=1 \
    RUN_ID="cloud_medium_${RUN_TIMESTAMP}" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE=1024 \
    \
    NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
    BIGRAM_VOCAB_SIZE=1536 BIGRAM_DIM=128 \
    XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
    VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10" \
    TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0 \
    MTP_NUM_HEADS=3 MTP_LOSS_WEIGHT=0.15 \
    Z_LOSS_WEIGHT=0.0001 QK_GAIN_INIT=1.5 \
    \
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_BACKEND_STEPS=5 \
    MUON_WD=0.04 ADAM_WD=0.04 \
    BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 \
    \
    ITERATIONS=500 WARMDOWN_ITERS=100 WARMUP_STEPS=10 \
    TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 \
    MAX_WALLCLOCK_SECONDS=0 \
    VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50 \
    EVAL_STRIDE=64 \
    \
    EMA_DECAY=0.997 \
    SWA_ENABLED=1 SWA_EVERY=50 \
    LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    TTT_ENABLED=0 \
    \
    SEED="${SEED:-1337}" \
    torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"
    ;;

  nottt)
    echo "[4/5] Full training (600s wallclock), TTT DISABLED — faster turnaround"
    NCCL_IB_DISABLE=1 \
    RUN_ID="cloud_nottt_${RUN_TIMESTAMP}" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE=1024 \
    \
    NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
    BIGRAM_VOCAB_SIZE=1536 BIGRAM_DIM=128 \
    XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
    VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10" \
    TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0 \
    MTP_NUM_HEADS=3 MTP_LOSS_WEIGHT=0.15 \
    Z_LOSS_WEIGHT=0.0001 QK_GAIN_INIT=1.5 \
    \
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_BACKEND_STEPS=5 \
    MUON_WD=0.04 ADAM_WD=0.04 \
    BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 \
    \
    ITERATIONS=20000 WARMDOWN_ITERS=3500 WARMUP_STEPS=20 \
    TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 \
    MAX_WALLCLOCK_SECONDS=600 \
    VAL_LOSS_EVERY=2000 TRAIN_LOG_EVERY=200 \
    EVAL_STRIDE=64 \
    \
    EMA_DECAY=0.997 \
    SWA_ENABLED=1 SWA_EVERY=50 \
    LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    TTT_ENABLED=0 \
    \
    SEED="${SEED:-1337}" \
    torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT" \
    2>&1 | tee "/workspace/cloud_nottt_${RUN_TIMESTAMP}.log"
    ;;

  full)
    # ---------------------------------------------------------------
    # FULL COMPETITION RUN — replicates 1.1194 BPB SOTA
    # All seeds: 1337, 42, 2025 (submit best; mean must beat SOTA)
    # ---------------------------------------------------------------
    SEED="${SEED:-1337}"
    LOG_FILE="/workspace/cloud_full_${RUN_TIMESTAMP}_seed${SEED}.log"
    echo "[5/5] FULL competition run:"
    echo "  Seed          : $SEED"
    echo "  Train budget  : 600s (MAX_WALLCLOCK_SECONDS)"
    echo "  TTT           : ENABLED (TTT_LR=0.002, TTT_EPOCHS=3, TTT_CHUNK_TOKENS=32768)"
    echo "  Target BPB    : ~1.1194 (SOTA 2026-03-23)"
    echo "  Log           : $LOG_FILE"
    echo ""

    NCCL_IB_DISABLE=1 \
    RUN_ID="cloud_full_${RUN_TIMESTAMP}_seed${SEED}" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE=1024 \
    \
    NUM_LAYERS=11 \
    MODEL_DIM=512 \
    NUM_HEADS=8 \
    NUM_KV_HEADS=4 \
    MLP_MULT=3 \
    \
    BIGRAM_VOCAB_SIZE=1536 \
    BIGRAM_DIM=128 \
    XSA_LAST_N=4 \
    ROPE_DIMS=16 \
    ROPE_BASE=10000.0 \
    LN_SCALE=1 \
    \
    VE_ENABLED=1 \
    VE_DIM=128 \
    VE_LAYERS="9,10" \
    \
    TIE_EMBEDDINGS=1 \
    LOGIT_SOFTCAP=30.0 \
    MTP_NUM_HEADS=3 \
    MTP_LOSS_WEIGHT=0.15 \
    Z_LOSS_WEIGHT=0.0001 \
    QK_GAIN_INIT=1.5 \
    \
    MATRIX_LR=0.025 \
    SCALAR_LR=0.025 \
    TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 \
    MUON_BACKEND_STEPS=5 \
    MUON_WD=0.04 \
    ADAM_WD=0.04 \
    BETA1=0.9 \
    BETA2=0.95 \
    ADAM_EPS=1e-8 \
    GRAD_CLIP_NORM=0.3 \
    \
    ITERATIONS=20000 \
    WARMDOWN_ITERS=3500 \
    WARMUP_STEPS=20 \
    TRAIN_BATCH_TOKENS=786432 \
    TRAIN_SEQ_LEN=2048 \
    MAX_WALLCLOCK_SECONDS=600 \
    VAL_LOSS_EVERY=2000 \
    TRAIN_LOG_EVERY=200 \
    EVAL_STRIDE=64 \
    \
    EMA_DECAY=0.997 \
    SWA_ENABLED=1 \
    SWA_EVERY=50 \
    LATE_QAT=1 \
    LATE_QAT_THRESHOLD=0.15 \
    \
    TTT_ENABLED=1 \
    TTT_LR=0.002 \
    TTT_EPOCHS=3 \
    TTT_CHUNK_TOKENS=32768 \
    TTT_FREEZE_BLOCKS=0 \
    TTT_MOMENTUM=0.9 \
    TTT_BATCH_SEQS=32 \
    TTT_GRAD_CLIP=1.0 \
    \
    SEED="$SEED" \
    torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT" \
    2>&1 | tee "$LOG_FILE"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash launch_cloud.sh [full|smoke|medium|nottt]"
    exit 1
    ;;
esac

# ---------------------------------------------------------------
# Results summary (full / nottt modes only)
# ---------------------------------------------------------------
if [ "$MODE" = "full" ] || [ "$MODE" = "nottt" ]; then
    echo ""
    echo "===== RESULTS SUMMARY ====="

    # Find the most recent log
    LATEST_LOG=$(ls -t /workspace/cloud_*.log 2>/dev/null | head -1 || echo "")
    if [ -n "$LATEST_LOG" ]; then
        echo "Log: $LATEST_LOG"
        echo ""
        echo "--- Key metrics ---"
        grep -E "val_bpb|stopping_early|final_int6|final_int8|legal_ttt|peak memory" \
             "$LATEST_LOG" 2>/dev/null || echo "(no log found)"
        echo ""
    fi

    echo "--- Artifact sizes ---"
    CODE_BYTES=$(wc -c < "$TRAIN_SCRIPT" 2>/dev/null || echo 0)
    echo "Code (train_gpt.py): $CODE_BYTES bytes"
    for f in final_model.int8.ptz final_model.int6.ptz; do
        if [ -f "$f" ]; then
            MODEL_BYTES=$(wc -c < "$f")
            TOTAL=$((CODE_BYTES + MODEL_BYTES))
            LIMIT=16000000
            STATUS="PASS"
            [ "$TOTAL" -ge "$LIMIT" ] && STATUS="FAIL"
            echo "$f: ${MODEL_BYTES} bytes  ->  total: ${TOTAL} bytes  [limit ${LIMIT}]  [$STATUS]"
        else
            echo "$f: NOT FOUND"
        fi
    done

    echo ""
    echo "--- Pre-TTT vs post-TTT BPB (from log) ---"
    if [ -n "$LATEST_LOG" ]; then
        PRE_BPB=$(grep "final_int6_sliding_window_s64_exact" "$LATEST_LOG" 2>/dev/null | \
                  tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
        POST_BPB=$(grep "legal_ttt_exact" "$LATEST_LOG" 2>/dev/null | \
                   tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A (TTT disabled)")
        echo "  Pre-TTT  (int6 sliding s64): $PRE_BPB"
        echo "  Post-TTT (legal TTT)       : $POST_BPB"
        echo "  SOTA target                : 1.11940000"
    fi

    echo ""
    echo "===== DONE ====="
    echo ""
    echo "Next steps:"
    echo "  1. If BPB matches (~1.1194): copy final_model.int6.ptz and $TRAIN_SCRIPT"
    echo "     to your records folder and open a PR."
    echo "  2. For 3-seed stat significance, re-run with SEED=42 and SEED=2025:"
    echo "     SEED=42   bash launch_cloud.sh full"
    echo "     SEED=2025 bash launch_cloud.sh full"
    echo "  3. Submission size check: code + model must be < 16,000,000 bytes."
fi
