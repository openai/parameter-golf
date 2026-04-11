#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"

if [[ ! -f "${repo_root}/.venv-wsl/bin/activate" ]]; then
  echo "Missing WSL venv at ${repo_root}/.venv-wsl" >&2
  echo "Set up the environment first." >&2
  exit 1
fi

source "${repo_root}/.venv-wsl/bin/activate"
cd "${repo_root}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

export DATA_PATH="${DATA_PATH:-./data/byte260_export/datasets/fineweb10B_byte260}"
export RUN_ID="${RUN_ID:-jepa_3080_30m}"
export SEED="${SEED:-1337}"

export ITERATIONS="${ITERATIONS:-200000}"
export WARMUP_STEPS="${WARMUP_STEPS:-3}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1200}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-400}"
export MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-131072}"
export FINAL_MAX_VAL_TOKENS="${FINAL_MAX_VAL_TOKENS:-131072}"
export FINAL_FULL_VAL="${FINAL_FULL_VAL:-0}"

export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export TARGET_SEQ_LEN="${TARGET_SEQ_LEN:-256}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

export MODEL_DIM="${MODEL_DIM:-384}"
export LATENT_DIM="${LATENT_DIM:-256}"
export ENCODER_LAYERS="${ENCODER_LAYERS:-8}"
export PREDICTOR_LAYERS="${PREDICTOR_LAYERS:-1}"
export NUM_HEADS="${NUM_HEADS:-8}"
export MLP_MULT="${MLP_MULT:-3}"
export MLP_ACTIVATION="${MLP_ACTIVATION:-relu_sq}"
export CONTEXT_POOL_STRIDE="${CONTEXT_POOL_STRIDE:-1}"
export PREDICT_CHUNK_LEN="${PREDICT_CHUNK_LEN:-32}"
export MEMORY_TOKENS="${MEMORY_TOKENS:-0}"
export MEMORY_LAYERS="${MEMORY_LAYERS:-0}"
export DECODER_LAYERS="${DECODER_LAYERS:-3}"
export PREDICTOR_RECUR_LAYERS="${PREDICTOR_RECUR_LAYERS:-0}"
export PREDICTOR_RECUR_PASSES="${PREDICTOR_RECUR_PASSES:-1}"
export DECODER_RECUR_LAYERS="${DECODER_RECUR_LAYERS:-0}"
export DECODER_RECUR_PASSES="${DECODER_RECUR_PASSES:-1}"
export LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-20}"

export CE_LOSS_WEIGHT="${CE_LOSS_WEIGHT:-1.0}"
export LATENT_LOSS_WEIGHT="${LATENT_LOSS_WEIGHT:-1.0}"
export SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.05}"
export SIGREG_COV_WEIGHT="${SIGREG_COV_WEIGHT:-0.05}"
export LEWM_STYLE="${LEWM_STYLE:-1}"
export DECODER_AUX_WEIGHT="${DECODER_AUX_WEIGHT:-1.0}"
export SIGREG_LAMBDA="${SIGREG_LAMBDA:-0.05}"
export CHUNK_LATENT_WEIGHT="${CHUNK_LATENT_WEIGHT:-0.25}"
export DECODER_BYTE_MASK_RATE="${DECODER_BYTE_MASK_RATE:-0.25}"
export EMA_DECAY="${EMA_DECAY:-0.998}"

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

export USE_COMPILE="${USE_COMPILE:-1}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"

export TTT_ENABLED="${TTT_ENABLED:-1}"
export TTT_LR="${TTT_LR:-0.0002}"
export TTT_EPOCHS="${TTT_EPOCHS:-2}"
export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
export TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"
export TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"
export TTT_TRAIN_MODE="${TTT_TRAIN_MODE:-late}"
export TTT_FREEZE_ENCODER_BLOCKS="${TTT_FREEZE_ENCODER_BLOCKS:-6}"
export TTT_LR_MIN_FRAC="${TTT_LR_MIN_FRAC:-0.2}"
export TTT_PURE_LATENT="${TTT_PURE_LATENT:-1}"

export BYTE_LOGIT_CACHE_ENABLED="${BYTE_LOGIT_CACHE_ENABLED:-1}"
export BYTE_LOGIT_CACHE_MIN_N="${BYTE_LOGIT_CACHE_MIN_N:-3}"
export BYTE_LOGIT_CACHE_MAX_N="${BYTE_LOGIT_CACHE_MAX_N:-8}"
export BYTE_LOGIT_CACHE_ALPHA="${BYTE_LOGIT_CACHE_ALPHA:-0.25}"
export BYTE_LOGIT_CACHE_MIN_COUNT="${BYTE_LOGIT_CACHE_MIN_COUNT:-2}"
export BYTE_LOGIT_CACHE_COUNT_SCALE="${BYTE_LOGIT_CACHE_COUNT_SCALE:-12}"

# Cross-decoder sweep
# Reference baseline (already run): jepa_3080_baseline val_bpb=1.9493 (8E+1P+3D, no cross-decoder)
#
# xdec_6e_4d  : 6E+1P+4XD ~20.3M params — fewer encoder layers, deeper cross-decoder
# xdec_7e_3d  : 7E+1P+3XD ~19.7M params — near-identical param count, cross-decoder
# xdec_6e_4d_knn : 6E+1P+4XD + latent k-NN cache (dtype bug now fixed)
configs=("xdec_6e_4d" "xdec_7e_3d" "xdec_6e_4d_knn")

for config in "${configs[@]}"; do
    export RUN_ID="jepa_3080_${config}"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1800}" # 30 mins per config

    # Shared defaults for this sweep
    export TARGET_PATCH_SIZE=1
    export LATENT_KNN_CACHE_ENABLED=0
    export LATENT_KNN_CACHE_K=3
    export LATENT_KNN_CACHE_ALPHA=0.1
    export TRAIN_SEQ_LEN=1024
    export TRAIN_BATCH_TOKENS=65536
    export VAL_BATCH_SIZE=65536
    export CROSS_DECODER=1

    if [[ "$config" == "xdec_6e_4d" || "$config" == "xdec_6e_4d_knn" ]]; then
        export ENCODER_LAYERS=6
        export DECODER_LAYERS=4
    fi

    if [[ "$config" == "xdec_7e_3d" ]]; then
        export ENCODER_LAYERS=7
        export DECODER_LAYERS=3
    fi

    if [[ "$config" == "xdec_6e_4d_knn" ]]; then
        export LATENT_KNN_CACHE_ENABLED=1
    fi

    echo "======================================================================"
    echo "Running ablation config: $config"
    echo "ENCODER_LAYERS=$ENCODER_LAYERS DECODER_LAYERS=$DECODER_LAYERS CROSS_DECODER=$CROSS_DECODER LATENT_KNN_CACHE_ENABLED=$LATENT_KNN_CACHE_ENABLED"
    echo "======================================================================"

    python ./jepa/train_gpt.py
done
