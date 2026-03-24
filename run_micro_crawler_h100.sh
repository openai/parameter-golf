#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# MICRO CRAWLER H100 TEST — 3flat + 1crawl×2, dim=800, trigram
# ═══════════════════════════════════════════════════════════════════════
#
# Optimal config from 167-run Spark sweep + 8-config micro crawler sweep:
#   3 flat blocks (unique, run once, clean gradients)
#   1 crawler block × 2 loops (shared, orthogonal fire)
#   = 5 effective depth, 4 stored blocks, dim=800
#
# Run on a single rented H100:
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

# ── Architecture: Micro Crawler 3f+1cx2 ──
export NUM_FLAT_LAYERS=3
export NUM_CRAWLER_LAYERS=1
export CRAWLER_LOOPS=2
export CRAWLER_MLP_MULT=4
export MODEL_DIM=800
export NUM_HEADS=10
export NUM_KV_HEADS=5
export MLP_MULT=4
export VOCAB_SIZE=1024

# ── Input conditioning ──
export TRIGRAM_VOCAB_SIZE=8192
export TRIGRAM_DIM=128

# ── Features ──
export XSA_LAST_N=1          # XSA on the crawler block (it's the only one)
export ROPE_DIMS=16           # partial RoPE
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="0"          # VE on crawler block 0
export TIE_EMBEDDINGS=1
export LOGIT_SOFTCAP=30.0

# ── Training ──
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=262144   # 256K — smaller batch = more steps on 1 GPU
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
export VAL_BATCH_SIZE=262144

# ── Run ID ──
export SEED=1337
export RUN_ID="micro_crawler_3f1cx2_d800_$(date +%Y%m%d_%H%M%S)"

echo "═══════════════════════════════════════════════════════════════════"
echo "MICRO CRAWLER H100 — 3flat + 1crawl×2 = 5 effective, dim=800"
echo "Run ID: $RUN_ID"
echo "═══════════════════════════════════════════════════════════════════"

# Estimate params: 4 blocks × 11 × 800² + 1024×800 + trigram ≈ 29.0M
echo "Estimated params: ~29M (4 stored blocks at dim=800, MLP 4x)"
echo "Expected artifact: ~15.6MB (int6+zstd)"
echo ""

python train_gpt_micro_crawler_h100.py

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "DONE — check logs/$RUN_ID.txt"
echo "═══════════════════════════════════════════════════════════════════"
