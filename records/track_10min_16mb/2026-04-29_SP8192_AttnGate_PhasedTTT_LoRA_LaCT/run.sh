#!/usr/bin/env bash
# run.sh — 2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT
# Target: 8×H100 RunPod  |  10-min train cap  |  10-min eval cap
# Score path: quantized + Multi-Phase Global SGD TTT (PR #1727 lineage)
# Default launch shape: one seed per run; repeat for 42, 314, 999 for competition results
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── 1. Install dependencies ────────────────────────────────────────────────
echo "[run.sh] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# flash-attn 3 wheel
FLASH_ATTN_WHEEL_INDEX="https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/"
if ! python3 -c "import sys; sys.path.insert(0, \"$SCRIPT_DIR\"); from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "[run.sh] FlashAttention not found; installing wheel..."
    python3 -m pip install --quiet flash_attn_3 --no-deps --find-links "$FLASH_ATTN_WHEEL_INDEX"
fi
python3 -c "import sys; sys.path.insert(0, \"$SCRIPT_DIR\"); from flash_attn_interface import flash_attn_func; print('[run.sh] FlashAttention backend ready.')"

# ── 2. Download / verify dataset ──────────────────────────────────────────
echo "[run.sh] Fetching FineWeb SP8192 dataset (128 train shards + val + tokenizer)..."
cd "$REPO_ROOT"
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

# ── 3. Environment / hyperparameters ──────────────────────────────────────
export DATA_DIR="$REPO_ROOT/data"

# Training caps
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export GPTQ_RESERVE_SECONDS="${GPTQ_RESERVE_SECONDS:-12}"

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

# Architecture features — GatedAttn (PR #1769, Qwen arXiv:2505.06708)
export GATED_ATTN_ENABLED=1         # input-dependent per-head sigmoid gate
export GATED_ATTN_INIT_STD=0.01     # init std for attn_gate_w
export GATED_ATTN_QUANT_GATE=1      # int8-per-row quantization for gate tensors (QuantGate)

# Phased TTT / LoRA adapter settings (PR #1727 path)
export TTT_LORA_ENABLED=0
export TTT_LORA_RANK="${TTT_LORA_RANK:-96}"
export TTT_LORA_LR="${TTT_LORA_LR:-0.0001}"
export TTT_CHUNK_SIZE="${TTT_CHUNK_SIZE:-48}"
export TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-64}"
export TTT_GRAD_STEPS="${TTT_GRAD_STEPS:-1}"
export TTT_WEIGHT_DECAY="${TTT_WEIGHT_DECAY:-0.5}"
export TTT_BETA1="${TTT_BETA1:-0.0}"
export TTT_BETA2="${TTT_BETA2:-0.999}"
export TTT_Q_LORA="${TTT_Q_LORA:-0}"           # no_qv mask: Q LoRA off (saves rank for K/MLP/O)
export TTT_V_LORA="${TTT_V_LORA:-0}"           # no_qv mask: V LoRA off
export TTT_K_LORA="${TTT_K_LORA:-1}"
export TTT_MLP_LORA="${TTT_MLP_LORA:-1}"
export TTT_O_LORA="${TTT_O_LORA:-1}"
export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adam}"
export PHASED_TTT_ENABLED="${PHASED_TTT_ENABLED:-1}"
export PHASED_TTT_PREFIX_DOCS="${PHASED_TTT_PREFIX_DOCS:-2000}"
export PHASED_TTT_NUM_PHASES="${PHASED_TTT_NUM_PHASES:-5}"
export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-2560}"
export GLOBAL_TTT_OPTIMIZER="${GLOBAL_TTT_OPTIMIZER:-adamw}"
export GLOBAL_TTT_LR="${GLOBAL_TTT_LR:-0.0008}"
export GLOBAL_TTT_WEIGHT_DECAY="${GLOBAL_TTT_WEIGHT_DECAY:-0.01}"
export GLOBAL_TTT_MOMENTUM="${GLOBAL_TTT_MOMENTUM:-0.9}"
export GLOBAL_TTT_EPOCHS="${GLOBAL_TTT_EPOCHS:-2}"
export GLOBAL_TTT_CHUNK_TOKENS="${GLOBAL_TTT_CHUNK_TOKENS:-32768}"
export GLOBAL_TTT_BATCH_SEQS="${GLOBAL_TTT_BATCH_SEQS:-32}"
export GLOBAL_TTT_WARMUP_START_LR="${GLOBAL_TTT_WARMUP_START_LR:-0.0}"
export GLOBAL_TTT_WARMUP_CHUNKS="${GLOBAL_TTT_WARMUP_CHUNKS:-2}"
export GLOBAL_TTT_GRAD_CLIP="${GLOBAL_TTT_GRAD_CLIP:-1.0}"
export GLOBAL_TTT_RESPECT_DOC_BOUNDARIES="${GLOBAL_TTT_RESPECT_DOC_BOUNDARIES:-1}"
export GPTQ_CALIBRATION_BATCHES="${GPTQ_CALIBRATION_BATCHES:-64}"

# Legacy TTT paths (disabled)
export TTT_ENABLED=1
export LACT_TTT_ENABLED=0

# Mixed GPTQ + LQER export (PR #1855)
export ARTIFACT_TARGET_BYTES=16000000
export MLP_CLIP_SIGMAS="${MLP_CLIP_SIGMAS:-12.0}"
export ATTN_CLIP_SIGMAS="${ATTN_CLIP_SIGMAS:-13.0}"
export LQER_ENABLED="${LQER_ENABLED:-1}"
export LQER_RANK="${LQER_RANK:-4}"
export LQER_TOP_K="${LQER_TOP_K:-3}"
export LQER_FACTOR_BITS="${LQER_FACTOR_BITS:-4}"
export LQER_ASYM_ENABLED="${LQER_ASYM_ENABLED:-1}"
export LQER_ASYM_GROUP="${LQER_ASYM_GROUP:-64}"

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
export SEED="${SEED:-42}"
echo "[run.sh] Starting torchrun with 8 GPUs..."
echo "[run.sh] MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS  SEED=$SEED"
echo "[run.sh] TTT_ENABLED=$TTT_ENABLED  PHASED_TTT_ENABLED=$PHASED_TTT_ENABLED  PHASES=$PHASED_TTT_NUM_PHASES"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_gpt.py

echo "[run.sh] Done. Artifact: $SCRIPT_DIR/final_model.int6.ptz"
echo "[run.sh] Logs:    $SCRIPT_DIR/logs/"
