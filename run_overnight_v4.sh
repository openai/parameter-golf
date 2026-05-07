#!/bin/bash
# Overnight v4 — 2026-03-22
# Purpose: Fill the critical gap — deeper models WITHOUT SWA at Int8
#
# Key findings from v3:
#   - SWA is DEAD — quantization-hostile in ALL cases (4 experiments confirm)
#   - WD=0.04 + OrthoInit + MLP3x + Int8 + slide64 = best Q BPB (1.4536)
#   - zstd saves ~40% size → massive headroom for deeper models
#   - 11L had best pre-quant val (1.4450 in exp36) but SWA ruined Q score
#
# Strategy: Test 10L and 11L with proven best config (no SWA), plus
#           zstd compression and BigramHash variants that fit budget.

set -euo pipefail

echo "=== Starting overnight v4 batch at $(date) ==="

# ─────────────────────────────────────────────────────────
# TIER 1: Deeper models with proven best config (NO SWA)
# These are the #1 priority untested direction
# ─────────────────────────────────────────────────────────

# Exp 39: 10 Layers + WD + OrthoInit + Int8 + zstd + slide64
# Expected: better than 9L (exp38), ~10-11 MB artifact
./run_experiment.sh exp39_10L_int8 \
  TRAIN_SCRIPT=train_gpt_exp.py \
  ITERATIONS=1000 TRAIN_BATCH_TOKENS=65536 \
  NUM_LAYERS=10 MLP_MULT=3 WEIGHT_DECAY=0.04 ORTHO_INIT=1 \
  COMPRESSOR=zstd \
  EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64 \
  VAL_LOSS_EVERY=200 WARMUP_STEPS=5

# Exp 40: 11 Layers + WD + OrthoInit + Int8 + zstd + slide64
# Expected: best pre-quant val, ~11-12 MB artifact
./run_experiment.sh exp40_11L_int8 \
  TRAIN_SCRIPT=train_gpt_exp.py \
  ITERATIONS=1000 TRAIN_BATCH_TOKENS=65536 \
  NUM_LAYERS=11 MLP_MULT=3 WEIGHT_DECAY=0.04 ORTHO_INIT=1 \
  COMPRESSOR=zstd \
  EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64 \
  VAL_LOSS_EVERY=200 WARMUP_STEPS=5

# Exp 41: 12 Layers — push the boundary (may be borderline on size)
# Expected: ~12-13 MB with zstd. If it fits, more depth = better
./run_experiment.sh exp41_12L_int8 \
  TRAIN_SCRIPT=train_gpt_exp.py \
  ITERATIONS=1000 TRAIN_BATCH_TOKENS=65536 \
  NUM_LAYERS=12 MLP_MULT=3 WEIGHT_DECAY=0.04 ORTHO_INIT=1 \
  COMPRESSOR=zstd \
  EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64 \
  VAL_LOSS_EVERY=200 WARMUP_STEPS=5

# ─────────────────────────────────────────────────────────
# TIER 2: Deeper models + BigramHash (smaller table to fit Int8)
# ─────────────────────────────────────────────────────────

# Exp 42: 10L + BigramHash-1024 + Int8 + zstd
# Smaller hash table (1024 vs 4096) to stay under 16 MB
./run_experiment.sh exp42_10L_bigram1k \
  TRAIN_SCRIPT=train_gpt_exp.py \
  ITERATIONS=1000 TRAIN_BATCH_TOKENS=65536 \
  NUM_LAYERS=10 MLP_MULT=3 WEIGHT_DECAY=0.04 ORTHO_INIT=1 \
  BIGRAM_HASH_SIZE=1024 \
  COMPRESSOR=zstd \
  EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64 \
  VAL_LOSS_EVERY=200 WARMUP_STEPS=5

# Exp 43: 9L + BigramHash-1024 + Int8 + zstd
# Same as best config (exp38) but with small bigram table
./run_experiment.sh exp43_9L_bigram1k \
  TRAIN_SCRIPT=train_gpt_exp.py \
  ITERATIONS=1000 TRAIN_BATCH_TOKENS=65536 \
  MLP_MULT=3 WEIGHT_DECAY=0.04 ORTHO_INIT=1 \
  BIGRAM_HASH_SIZE=1024 \
  COMPRESSOR=zstd \
  EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64 \
  VAL_LOSS_EVERY=200 WARMUP_STEPS=5

# ─────────────────────────────────────────────────────────
# TIER 3: Deeper models + QAT Int6 (fallback if Int8 too big)
# Without SWA this time — SWA was the problem, not QAT
# ─────────────────────────────────────────────────────────

# Exp 44: 11L + QAT Int6 + zstd (NO SWA)
# Re-test Int6 path without SWA poisoning the weights
./run_experiment.sh exp44_11L_qat_noswa \
  TRAIN_SCRIPT=train_gpt_exp.py \
  ITERATIONS=1000 TRAIN_BATCH_TOKENS=65536 \
  NUM_LAYERS=11 MLP_MULT=3 WEIGHT_DECAY=0.04 ORTHO_INIT=1 \
  QAT=1 QUANT_BITS=6 COMPRESSOR=zstd \
  EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64 \
  VAL_LOSS_EVERY=200 WARMUP_STEPS=5

# Exp 45: 12L + QAT Int6 + zstd (NO SWA)
# If 12L is too big for Int8, Int6+zstd should fit easily
./run_experiment.sh exp45_12L_qat_noswa \
  TRAIN_SCRIPT=train_gpt_exp.py \
  ITERATIONS=1000 TRAIN_BATCH_TOKENS=65536 \
  NUM_LAYERS=12 MLP_MULT=3 WEIGHT_DECAY=0.04 ORTHO_INIT=1 \
  QAT=1 QUANT_BITS=6 COMPRESSOR=zstd \
  EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64 \
  VAL_LOSS_EVERY=200 WARMUP_STEPS=5

# ─────────────────────────────────────────────────────────
# TIER 4: Warmdown tuning (refinement)
# Default WARMDOWN_ITERS=1200 tuned for 20K steps.
# At 1000 local steps, try shorter warmdown.
# ─────────────────────────────────────────────────────────

# Exp 46: Best config + warmdown=200 (20% of 1000 steps)
./run_experiment.sh exp46_warmdown200 \
  TRAIN_SCRIPT=train_gpt_exp.py \
  ITERATIONS=1000 TRAIN_BATCH_TOKENS=65536 \
  MLP_MULT=3 WEIGHT_DECAY=0.04 ORTHO_INIT=1 \
  WARMDOWN_ITERS=200 \
  COMPRESSOR=zstd \
  EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64 \
  VAL_LOSS_EVERY=200 WARMUP_STEPS=5

echo ""
echo "=== All v4 experiments completed at $(date) ==="
echo ""
echo "Run 'python3 compare_experiments.py' to see results."
