#!/usr/bin/env bash
# run.sh — 2026-04-20_SP8192_AttnGate_MultiPhaseTTT_LaCT
# Target: 8×H100 RunPod  |  10-min train cap  |  10-min eval cap
# Score path: quantized + multi-phase TTT (MULTIPHASE_TTT_ENABLED=1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── 1. Install dependencies ────────────────────────────────────────────────
echo "[run.sh] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# flash-attn 3 wheel — must match the April 9 record path exactly
FLASH_ATTN_WHEEL_INDEX="https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/"
if ! python3 -c "from flash_attn.flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "[run.sh] FlashAttention not found; installing April 9 wheel path..."
    python3 -m pip install --quiet flash_attn_3 --no-deps --find-links "$FLASH_ATTN_WHEEL_INDEX"
fi
python3 -c "from flash_attn.flash_attn_interface import flash_attn_func; print('[run.sh] FlashAttention backend ready.')"

# ── 2. Download / verify dataset ──────────────────────────────────────────
echo "[run.sh] Fetching FineWeb SP8192 dataset (128 train shards + val + tokenizer)..."
cd "$REPO_ROOT"
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

# ── 3. Environment / hyperparameters ──────────────────────────────────────
# Training caps
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export GPTQ_RESERVE_SECONDS="${GPTQ_RESERVE_SECONDS:-12}"

# Reproducibility
export SEED="${SEED:-42}"

# Architecture — must match accepted baseline
export VOCAB_SIZE=8192
export NUM_LAYERS=11
export MODEL_DIM=512
export EMBEDDING_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export QK_GAIN_INIT=5.25
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35
export PARALLEL_RESIDUAL_START=7

# New features for this record
export ATTN_GATE_ENABLED=1          # zero-init per-channel gate on attn output
export MULTIPHASE_TTT_ENABLED=1     # 4-phase score-first TTT (primary scored path)
export TTT_ENABLED=0                # single-phase legacy TTT (disabled)
export LACT_TTT_ENABLED=0           # LaCT adapter (disabled by default; enable to explore)

# Multi-phase TTT boundaries
export TTT_PHASE_A_END=0.00         # Phase A disabled by default in this profile
export TTT_PHASE_B_END=0.80         # Phase B: 0–80%  (all params, full LR)
export TTT_PHASE_C_END=0.95         # Phase C: 80–95% (all params, 1.0× LR)
# Phase D: 95–100%                  # (all params, 0.5× LR)
export TTT_PHASE_C_LR_SCALE=1.0
export TTT_PHASE_D_LR_SCALE=0.5

# Entropy GPTQ allocator
export EXPORT_ALLOCATOR=entropy
export ARTIFACT_TARGET_BYTES=16000000
export ALLOCATOR_ATTN_BITS="6,7"   # conservative: no 5-bit on attention
export ALLOCATOR_MATRIX_BITS="5,6,7"
export ALLOCATOR_EMBED_BITS="8"

# Training schedule
export ITERATIONS=20000
export WARMDOWN_FRAC=0.72
export TRAIN_BATCH_TOKENS=786432
export TRAIN_LOG_EVERY=500
export VAL_LOSS_EVERY=4000

# EMA / optimiser
export EMA_DECAY=0.9965
export MUON_WD=0.095

# ── 4. Launch ──────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
echo "[run.sh] Starting torchrun with 8 GPUs..."
echo "[run.sh] MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS  SEED=$SEED"
echo "[run.sh] ATTN_GATE_ENABLED=$ATTN_GATE_ENABLED  MULTIPHASE_TTT_ENABLED=$MULTIPHASE_TTT_ENABLED"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_gpt.py

echo "[run.sh] Done. Artifact: $SCRIPT_DIR/final_model.int6.ptz"
echo "[run.sh] Logs:    $SCRIPT_DIR/logs/"
