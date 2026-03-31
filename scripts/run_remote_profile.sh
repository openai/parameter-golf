#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${1:-base10l}"
shift || true

case "$PROFILE" in
  base10l)
    export RUN_ID="${RUN_ID:-base10l}"
    ;;
  zloss_low)
    export RUN_ID="${RUN_ID:-zloss_low}"
    export Z_LOSS_COEF="${Z_LOSS_COEF:-0.0001}"
    ;;
  zloss_med)
    export RUN_ID="${RUN_ID:-zloss_med}"
    export Z_LOSS_COEF="${Z_LOSS_COEF:-0.0003}"
    ;;
  twice_low)
    export RUN_ID="${RUN_ID:-twice_low}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    ;;
  twice_layerwise)
    export RUN_ID="${RUN_ID:-twice_layerwise}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ATTN_TWICE_ALPHA_SLOPE="${ATTN_TWICE_ALPHA_SLOPE:-0.5}"
    ;;
  zloss_twice)
    export RUN_ID="${RUN_ID:-zloss_twice}"
    export Z_LOSS_COEF="${Z_LOSS_COEF:-0.0001}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    ;;
  eval2048)
    export RUN_ID="${RUN_ID:-eval2048}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-2048}"
    ;;
  twice_eval2048_ttt1024)
    export RUN_ID="${RUN_ID:-twice_eval2048_ttt1024}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    ;;
  drope_eval)
    export RUN_ID="${RUN_ID:-drope_eval}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export ROUNDTRIP_ROPE_SCALING="${ROUNDTRIP_ROPE_SCALING:-drope}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    ;;
  yarn_eval)
    export RUN_ID="${RUN_ID:-yarn_eval}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export ROUNDTRIP_ROPE_SCALING="${ROUNDTRIP_ROPE_SCALING:-yarn}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    ;;
  mtp_low)
    export RUN_ID="${RUN_ID:-mtp_low}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export MTP_DEPTH="${MTP_DEPTH:-2}"
    export MTP_LOSS_WEIGHT="${MTP_LOSS_WEIGHT:-0.1}"
    ;;
  muon_balance)
    export RUN_ID="${RUN_ID:-muon_balance}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export MUON_UPDATE_BALANCE="${MUON_UPDATE_BALANCE:-0.5}"
    ;;
  hybrid_delta)
    export RUN_ID="${RUN_ID:-hybrid_delta}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export HYBRID_DELTA_EVERY="${HYBRID_DELTA_EVERY:-4}"
    ;;
  shared_depth)
    export RUN_ID="${RUN_ID:-shared_depth}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export SHARED_DEPTH_N="${SHARED_DEPTH_N:-4}"
    export SHARED_DEPTH_GAIN="${SHARED_DEPTH_GAIN:-0.15}"
    ;;
  shared_depth_midshare)
    export RUN_ID="${RUN_ID:-shared_depth_midshare}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export SHARED_DEPTH_N="${SHARED_DEPTH_N:-4}"
    export SHARED_DEPTH_GAIN="${SHARED_DEPTH_GAIN:-0.05}"
    export SHARED_DEPTH_EDGE_UNIQUE="${SHARED_DEPTH_EDGE_UNIQUE:-2}"
    ;;
  copycore_v1)
    export RUN_ID="${RUN_ID:-copycore_v1}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export MLP_MULT="${MLP_MULT:-3}"
    export MLP_ACT="${MLP_ACT:-leaky2}"
    export LEAKY_RELU_SLOPE="${LEAKY_RELU_SLOPE:-0.5}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
    export EMA_DECAY="${EMA_DECAY:-0.997}"
    export SWA_START_FRAC="${SWA_START_FRAC:-0.6}"
    export SWA_STRIDE="${SWA_STRIDE:-50}"
    ;;
  winner_locked)
    export RUN_ID="${RUN_ID:-winner_locked}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    ;;
  winner_ema_swa)
    export RUN_ID="${RUN_ID:-winner_ema_swa}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export EMA_DECAY="${EMA_DECAY:-0.997}"
    export SWA_START_FRAC="${SWA_START_FRAC:-0.6}"
    export SWA_STRIDE="${SWA_STRIDE:-50}"
    ;;
  winner_wd03)
    export RUN_ID="${RUN_ID:-winner_wd03}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.03}"
    ;;
  winner_wd04)
    export RUN_ID="${RUN_ID:-winner_wd04}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    ;;
  winner_warm3500)
    export RUN_ID="${RUN_ID:-winner_warm3500}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
    ;;
  winner_lr18)
    export RUN_ID="${RUN_ID:-winner_lr18}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MATRIX_LR="${MATRIX_LR:-0.018}"
    ;;
  winner_wd03_ema)
    export RUN_ID="${RUN_ID:-winner_wd03_ema}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.03}"
    export EMA_DECAY="${EMA_DECAY:-0.997}"
    export SWA_START_FRAC="${SWA_START_FRAC:-0.6}"
    export SWA_STRIDE="${SWA_STRIDE:-50}"
    ;;
  winner_mlp3)
    export RUN_ID="${RUN_ID:-winner_mlp3}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MLP_MULT="${MLP_MULT:-3}"
    ;;
  wd04_locked)
    export RUN_ID="${RUN_ID:-wd04_locked}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    ;;
  wd04_warm3500)
    export RUN_ID="${RUN_ID:-wd04_warm3500}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
    ;;
  wd04_warm4000)
    export RUN_ID="${RUN_ID:-wd04_warm4000}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}"
    ;;
  wd04_lr18)
    export RUN_ID="${RUN_ID:-wd04_lr18}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export MATRIX_LR="${MATRIX_LR:-0.018}"
    ;;
  wd04_lr19)
    export RUN_ID="${RUN_ID:-wd04_lr19}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export MATRIX_LR="${MATRIX_LR:-0.019}"
    ;;
  wd04_scalar18)
    export RUN_ID="${RUN_ID:-wd04_scalar18}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export SCALAR_LR="${SCALAR_LR:-0.018}"
    ;;
  wd04_tied28)
    export RUN_ID="${RUN_ID:-wd04_tied28}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export TIED_EMBED_LR="${TIED_EMBED_LR:-0.028}"
    ;;
  wd04_ema_swa)
    export RUN_ID="${RUN_ID:-wd04_ema_swa}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export EMA_DECAY="${EMA_DECAY:-0.997}"
    export SWA_START_FRAC="${SWA_START_FRAC:-0.6}"
    export SWA_STRIDE="${SWA_STRIDE:-50}"
    ;;
  wd04_warm3500_lr18)
    export RUN_ID="${RUN_ID:-wd04_warm3500_lr18}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
    export MATRIX_LR="${MATRIX_LR:-0.018}"
    ;;
  wd04_zloss)
    export RUN_ID="${RUN_ID:-wd04_zloss}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export Z_LOSS_COEF="${Z_LOSS_COEF:-0.0001}"
    ;;
  hunt_locked_best)
    export RUN_ID="${RUN_ID:-hunt_locked_best}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    ;;
  hunt_11l_wd04)
    export RUN_ID="${RUN_ID:-hunt_11l_wd04}"
    export NUM_LAYERS="${NUM_LAYERS:-11}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    ;;
  hunt_11l_wd04_warm4000)
    export RUN_ID="${RUN_ID:-hunt_11l_wd04_warm4000}"
    export NUM_LAYERS="${NUM_LAYERS:-11}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}"
    ;;
  hunt_11l_mlp3_leaky)
    export RUN_ID="${RUN_ID:-hunt_11l_mlp3_leaky}"
    export NUM_LAYERS="${NUM_LAYERS:-11}"
    export MLP_MULT="${MLP_MULT:-3}"
    export MLP_ACT="${MLP_ACT:-leaky2}"
    export LEAKY_RELU_SLOPE="${LEAKY_RELU_SLOPE:-0.5}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}"
    ;;
  hunt_11l_mlp3_leaky_ema)
    export RUN_ID="${RUN_ID:-hunt_11l_mlp3_leaky_ema}"
    export NUM_LAYERS="${NUM_LAYERS:-11}"
    export MLP_MULT="${MLP_MULT:-3}"
    export MLP_ACT="${MLP_ACT:-leaky2}"
    export LEAKY_RELU_SLOPE="${LEAKY_RELU_SLOPE:-0.5}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}"
    export EMA_DECAY="${EMA_DECAY:-0.997}"
    export SWA_START_FRAC="${SWA_START_FRAC:-0.6}"
    export SWA_STRIDE="${SWA_STRIDE:-50}"
    ;;
  hunt_11l_layerwise)
    export RUN_ID="${RUN_ID:-hunt_11l_layerwise}"
    export NUM_LAYERS="${NUM_LAYERS:-11}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ATTN_TWICE_ALPHA_SLOPE="${ATTN_TWICE_ALPHA_SLOPE:-0.5}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    ;;
  hunt_11l_fullheads)
    export RUN_ID="${RUN_ID:-hunt_11l_fullheads}"
    export NUM_LAYERS="${NUM_LAYERS:-11}"
    export NUM_KV_HEADS="${NUM_KV_HEADS:-8}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    ;;
  hunt_11l_untied)
    export RUN_ID="${RUN_ID:-hunt_11l_untied}"
    export NUM_LAYERS="${NUM_LAYERS:-11}"
    export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-0}"
    export HEAD_LR="${HEAD_LR:-0.008}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    ;;
  hunt_shared11_mid)
    export RUN_ID="${RUN_ID:-hunt_shared11_mid}"
    export NUM_LAYERS="${NUM_LAYERS:-11}"
    export SHARED_DEPTH_N="${SHARED_DEPTH_N:-5}"
    export SHARED_DEPTH_EDGE_UNIQUE="${SHARED_DEPTH_EDGE_UNIQUE:-2}"
    export SHARED_DEPTH_GAIN="${SHARED_DEPTH_GAIN:-0.05}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    ;;
  hunt_11l_mtp)
    export RUN_ID="${RUN_ID:-hunt_11l_mtp}"
    export NUM_LAYERS="${NUM_LAYERS:-11}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    export ROUNDTRIP_EVAL_SEQ_LEN="${ROUNDTRIP_EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
    export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.04}"
    export MTP_DEPTH="${MTP_DEPTH:-2}"
    export MTP_LOSS_WEIGHT="${MTP_LOSS_WEIGHT:-0.05}"
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    echo "Profiles: base10l zloss_low zloss_med twice_low twice_layerwise zloss_twice eval2048 twice_eval2048_ttt1024 drope_eval yarn_eval mtp_low muon_balance hybrid_delta shared_depth shared_depth_midshare copycore_v1 winner_locked winner_ema_swa winner_wd03 winner_wd04 winner_warm3500 winner_lr18 winner_wd03_ema winner_mlp3 wd04_locked wd04_warm3500 wd04_warm4000 wd04_lr18 wd04_lr19 wd04_scalar18 wd04_tied28 wd04_ema_swa wd04_warm3500_lr18 wd04_zloss hunt_locked_best hunt_11l_wd04 hunt_11l_wd04_warm4000 hunt_11l_mlp3_leaky hunt_11l_mlp3_leaky_ema hunt_11l_layerwise hunt_11l_fullheads hunt_11l_untied hunt_shared11_mid hunt_11l_mtp" >&2
    exit 1
    ;;
esac

bash scripts/run_remote_experiment.sh "$@"
