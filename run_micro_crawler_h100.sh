#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# MICRO CRAWLER H100 TEST — 4flat + 2crawl×2, dim=640, trigram
# ═══════════════════════════════════════════════════════════════════════
#
# Balanced micro crawler architecture:
#   4 flat blocks (unique, run once, clean gradients)
#   2 crawler blocks × 2 loops (shared pair, orthogonal double-tap)
#   = 8 effective depth, 6 stored blocks, dim=640
#   F = C×L → 50/50 balanced split
#
# Run on 8xH100 SXM:
#   chmod +x run_micro_crawler_h100.sh
#   ./run_micro_crawler_h100.sh
#
# Prerequisites on the remote machine:
#   - CUDA + PyTorch with flash_attn_interface
#   - pip install sentencepiece zstandard
#   - Data in ./data/datasets/fineweb10B_sp1024/
#   - Tokenizer in ./data/tokenizers/fineweb_1024_bpe.model
#
set -euo pipefail

# ── Architecture: Micro Crawler 4f+2cx2 (balanced) ──
export NUM_FLAT_LAYERS=4
export NUM_CRAWLER_LAYERS=2
export CRAWLER_LOOPS=2
export CRAWLER_MLP_MULT=4
export CRAWLER_CADENCE=5            # N/N/N/N/C — 4 clean steps then 1 crawl
export CRAWLER_CADENCE_OFFSET=4
export MODEL_DIM=640
export NUM_HEADS=10
export NUM_KV_HEADS=5
export MLP_MULT=4
export VOCAB_SIZE=1024

# ── Input conditioning ──
export TRIGRAM_VOCAB_SIZE=8192
export TRIGRAM_DIM=128

# ── Features ──
export XSA_LAST_N=2          # XSA on both crawler blocks
export ROPE_DIMS=16           # partial RoPE
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="0,1"        # VE on both crawler blocks
export TIE_EMBEDDINGS=1
export LOGIT_SOFTCAP=30.0

# ── Training ──
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432    # full batch for 8xH100
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600
export WARMUP_STEPS=20
export WARMDOWN_ITERS=2500         # shorter warmdown for 1-GPU step budget
export GRAD_CLIP_NORM=0.3

# ── Optimizer ──
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export TIED_EMBED_INIT_STD=0.005
export MUON_MOMENTUM=0.99
export MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04
export ADAM_WD=0.04
export MUON_BETA2=0.95

# ── EMA / SWA / QAT ──
export SWA_ENABLED=1
export SWA_EVERY=50
export QAT_ENABLED=0
export LATE_QAT_THRESHOLD=0.15

# ── Late-stage: TTT burst + self-distillation ──
export TTT_BURST_ENABLED=1
export TTT_BURST_EPOCHS=2
export TTT_BURST_LR_FACTOR=0.1
export TTT_BURST_STEPS=100
export TTT_BURST_TRIGGER=0.05
export DISTILL_ENABLED=1
export DISTILL_STEPS=50
export DISTILL_LR_FACTOR=0.05
export DISTILL_TEMPERATURE=2.0
export DISTILL_ALPHA=0.7

# ── Eval ──
export EVAL_STRIDE=64
export VAL_LOSS_EVERY=500
export VAL_BATCH_SIZE=524288

# ── Run ID ──
export SEED=1337
export RUN_ID="micro_crawler_3f1cx2_d800_$(date +%Y%m%d_%H%M%S)"

echo "═══════════════════════════════════════════════════════════════════"
echo "MICRO CRAWLER H100 — 4flat + 2crawl×2 = 8 effective, dim=640"
echo "Run ID: $RUN_ID"
echo "═══════════════════════════════════════════════════════════════════"

# Estimate params: 6 blocks × 11 × 640² + 1024×640 + trigram ≈ 28.0M
echo "Estimated params: ~28M (6 stored blocks at dim=640, MLP 4x)"
echo "Expected artifact: ~15.1MB (int6+zstd)"
echo ""

torchrun --nproc_per_node=8 train_gpt_micro_crawler_h100.py

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "DONE — check logs/$RUN_ID.txt"
echo "═══════════════════════════════════════════════════════════════════"
