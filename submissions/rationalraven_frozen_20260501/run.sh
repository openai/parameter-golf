#!/usr/bin/env bash
# Train RationalRaven (sp4096 + MLP×3.25 + LeakyReLU² + late-QAT int8 attn/KV) from scratch.
# Reproduces the recipe that produced rationalraven_final.mixed.ptz / val_bpb=1.139722 (seed 1339).
#
# Requirements before running:
#   - 8×H100 SXM (or compatible) on a single node
#   - PyTorch 2.x with FlashAttention 2 build
#   - sentencepiece, numpy, huggingface-hub, zstandard
#   - FineWeb sp4096 shards reachable via either:
#       (a) ./data/datasets/fineweb10B_sp4096/  +  ./data/tokenizers/fineweb_4096_bpe.model, or
#       (b) HF auth set; train_gpt.py will pull from MATCHED_FINEWEB_REPO_ID
#
# To override (e.g. seed, wallclock cap, output dir): export the var before invoking.

set -euo pipefail

# ── RationalRaven binding env ─────────────────────
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-2000}"
export MATRIX_LR="${MATRIX_LR:-0.025}"
export SCALAR_LR="${SCALAR_LR:-0.025}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"

export MLP_ACTIVATION="${MLP_ACTIVATION:-leaky_relu2}"
export LEAKY_RELU_SLOPE="${LEAKY_RELU_SLOPE:-0.5}"

export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.3}"
export LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-18}"

export NUM_LAYERS="${NUM_LAYERS:-11}"
export MLP_MULT="${MLP_MULT:-3.25}"

export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-500}"

export LATE_QAT_STEPS="${LATE_QAT_STEPS:-2400}"
# QAT_START_STEP pins the late-QAT trigger to an explicit step (deterministic,
# immune to first-step torch.compile cold-start poisoning the wallclock estimate).
# 6048 = 8448 banked total steps - 2400 LATE_QAT_STEPS (matches banked artifact).
export QAT_START_STEP="${QAT_START_STEP:-6048}"
export EMA_DECAY="${EMA_DECAY:-0.997}"
export EMA_FP32="${EMA_FP32:-1}"

export WEIGHT_DECAY="${WEIGHT_DECAY:-0.04}"
export EVAL_STRIDE="${EVAL_STRIDE:-0}"
export USE_DELTA_FROM_INIT="${USE_DELTA_FROM_INIT:-1}"

export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

export QUANT_SCALE_SCHEME="${QUANT_SCALE_SCHEME:-per_row}"
export QUANT_INT8_CATS="${QUANT_INT8_CATS:-.attn.proj,.c_k,.c_v}"

export VOCAB_SIZE="${VOCAB_SIZE:-4096}"
export DATA_PATH="${DATA_PATH:-data/datasets/fineweb10B_sp4096}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizers/fineweb_4096_bpe.model}"
export MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-kevclark/parameter-golf}"

export SEED="${SEED:-1339}"

# ── Launch ────────────────────────────────────────
NPROC="${NPROC_PER_NODE:-8}"

echo "Training RationalRaven with SEED=${SEED}, NPROC=${NPROC}, MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
exec torchrun --standalone --nproc_per_node="${NPROC}" "$(dirname "$0")/train_gpt.py"
