#!/usr/bin/env bash
# Single H100 / A100 run for Google Colab — 10-minute test.
#
# HOW TO RUN ON GOOGLE COLAB:
# ─────────────────────────────────────────────────────────────────────────────
# 1. Open a new Colab notebook (Runtime → Change runtime type → H100 or A100)
#
# 2. In a code cell, clone the repo and install deps:
#
#    !git clone https://github.com/openai/parameter-golf.git  # or your fork
#    %cd parameter-golf
#    !pip install -q -r requirements.txt
#
# 3. Download the data (~4 GB, takes ~5-10 min):
#
#    !python data/cached_challenge_fineweb.py --variant byte260
#
# 4. Run this script:
#
#    !bash run_h100_colab.sh 2>&1 | tee logs/h100_colab.txt
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

export DATA_PATH="${DATA_PATH:-./data/byte260_export/datasets/fineweb10B_byte260}"
export RUN_ID="${RUN_ID:-jepa_h100_xdec_ctx1536}"
export SEED="${SEED:-1337}"

# ── Wallclock ─────────────────────────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"   # 10 minutes
export ITERATIONS="${ITERATIONS:-200000}"
export WARMUP_STEPS="${WARMUP_STEPS:-10}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1200}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-131072}"
export FINAL_MAX_VAL_TOKENS="${FINAL_MAX_VAL_TOKENS:-131072}"
export FINAL_FULL_VAL="${FINAL_FULL_VAL:-0}"

# ── Sequence / batch ──────────────────────────────────────────────────────────
# H100 has 80 GB — use longer context and bigger batch than 3080.
# seq_len=1536 → context=1280, target=256
# batch=196608 = 32 microbatch × 4 grad_accum × 1536 seq_len
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1536}"
export TARGET_SEQ_LEN="${TARGET_SEQ_LEN:-256}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-196608}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-196608}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

# ── Model architecture ────────────────────────────────────────────────────────
# 6E + 1P + 4 CrossDecoder  ≈ 20.3M params (fits ≤ 16 MiB int8+zlib)
export MODEL_DIM="${MODEL_DIM:-384}"
export LATENT_DIM="${LATENT_DIM:-256}"
export ENCODER_LAYERS="${ENCODER_LAYERS:-6}"
export PREDICTOR_LAYERS="${PREDICTOR_LAYERS:-1}"
export DECODER_LAYERS="${DECODER_LAYERS:-4}"
export CROSS_DECODER="${CROSS_DECODER:-1}"
export NUM_HEADS="${NUM_HEADS:-8}"
export MLP_MULT="${MLP_MULT:-3}"
export MLP_ACTIVATION="${MLP_ACTIVATION:-relu_sq}"
export CONTEXT_POOL_STRIDE="${CONTEXT_POOL_STRIDE:-1}"
export PREDICT_CHUNK_LEN="${PREDICT_CHUNK_LEN:-32}"
export MEMORY_TOKENS="${MEMORY_TOKENS:-0}"
export MEMORY_LAYERS="${MEMORY_LAYERS:-0}"
export TARGET_PATCH_SIZE="${TARGET_PATCH_SIZE:-1}"
export LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-20}"
export PREDICTOR_RECUR_LAYERS="${PREDICTOR_RECUR_LAYERS:-0}"
export PREDICTOR_RECUR_PASSES="${PREDICTOR_RECUR_PASSES:-1}"
export DECODER_RECUR_LAYERS="${DECODER_RECUR_LAYERS:-0}"
export DECODER_RECUR_PASSES="${DECODER_RECUR_PASSES:-1}"

# ── Loss weights ──────────────────────────────────────────────────────────────
# Reduce LATENT_LOSS_WEIGHT to 0.5 to stabilise training at longer context.
# Previous 1536-context run failed with weight=1.0 (latent loss blew up).
export CE_LOSS_WEIGHT="${CE_LOSS_WEIGHT:-1.0}"
export LATENT_LOSS_WEIGHT="${LATENT_LOSS_WEIGHT:-0.5}"
export SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.05}"
export SIGREG_COV_WEIGHT="${SIGREG_COV_WEIGHT:-0.05}"
export LEWM_STYLE="${LEWM_STYLE:-1}"
export DECODER_AUX_WEIGHT="${DECODER_AUX_WEIGHT:-1.0}"
export SIGREG_LAMBDA="${SIGREG_LAMBDA:-0.05}"
export CHUNK_LATENT_WEIGHT="${CHUNK_LATENT_WEIGHT:-0.25}"
export DECODER_BYTE_MASK_RATE="${DECODER_BYTE_MASK_RATE:-0.25}"
export EMA_DECAY="${EMA_DECAY:-0.998}"

# ── Optimizer ─────────────────────────────────────────────────────────────────
export EMBED_LR="${EMBED_LR:-0.25}"
export MATRIX_LR="${MATRIX_LR:-0.035}"
export SCALAR_LR="${SCALAR_LR:-0.02}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-500}"
export BETA1="${BETA1:-0.9}"
export BETA2="${BETA2:-0.95}"
export ADAM_EPS="${ADAM_EPS:-1e-8}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

# ── Compile ───────────────────────────────────────────────────────────────────
export USE_COMPILE="${USE_COMPILE:-1}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"

# ── TTT ───────────────────────────────────────────────────────────────────────
export TTT_ENABLED="${TTT_ENABLED:-1}"
export TTT_LR="${TTT_LR:-0.0002}"
export TTT_EPOCHS="${TTT_EPOCHS:-2}"
export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
export TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"
export TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"
export TTT_TRAIN_MODE="${TTT_TRAIN_MODE:-late}"
export TTT_FREEZE_ENCODER_BLOCKS="${TTT_FREEZE_ENCODER_BLOCKS:-4}"
export TTT_LR_MIN_FRAC="${TTT_LR_MIN_FRAC:-0.2}"
export TTT_PURE_LATENT="${TTT_PURE_LATENT:-1}"

# ── Byte logit cache ──────────────────────────────────────────────────────────
export BYTE_LOGIT_CACHE_ENABLED="${BYTE_LOGIT_CACHE_ENABLED:-1}"
export BYTE_LOGIT_CACHE_MIN_N="${BYTE_LOGIT_CACHE_MIN_N:-3}"
export BYTE_LOGIT_CACHE_MAX_N="${BYTE_LOGIT_CACHE_MAX_N:-8}"
export BYTE_LOGIT_CACHE_ALPHA="${BYTE_LOGIT_CACHE_ALPHA:-0.25}"
export BYTE_LOGIT_CACHE_MIN_COUNT="${BYTE_LOGIT_CACHE_MIN_COUNT:-2}"
export BYTE_LOGIT_CACHE_COUNT_SCALE="${BYTE_LOGIT_CACHE_COUNT_SCALE:-12}"

# ── Latent k-NN cache ─────────────────────────────────────────────────────────
export LATENT_KNN_CACHE_ENABLED="${LATENT_KNN_CACHE_ENABLED:-0}"
export LATENT_KNN_CACHE_K="${LATENT_KNN_CACHE_K:-3}"
export LATENT_KNN_CACHE_ALPHA="${LATENT_KNN_CACHE_ALPHA:-0.1}"

echo "======================================================================"
echo "H100 Colab run: $RUN_ID"
echo "seq_len=$TRAIN_SEQ_LEN  batch=$TRAIN_BATCH_TOKENS  wallclock=${MAX_WALLCLOCK_SECONDS}s"
echo "encoder_layers=$ENCODER_LAYERS  decoder_layers=$DECODER_LAYERS  cross_decoder=$CROSS_DECODER"
echo "latent_loss_weight=$LATENT_LOSS_WEIGHT"
echo "======================================================================"

python ./jepa/train_gpt.py
