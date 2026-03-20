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
  *)
    echo "Unknown profile: $PROFILE" >&2
    echo "Profiles: base10l zloss_low zloss_med twice_low twice_layerwise zloss_twice eval2048 twice_eval2048_ttt1024 drope_eval yarn_eval mtp_low muon_balance hybrid_delta" >&2
    exit 1
    ;;
esac

bash scripts/run_remote_experiment.sh "$@"
