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
  zloss_twice)
    export RUN_ID="${RUN_ID:-zloss_twice}"
    export Z_LOSS_COEF="${Z_LOSS_COEF:-0.0001}"
    export ATTN_TWICE_ALPHA="${ATTN_TWICE_ALPHA:-0.05}"
    ;;
  eval2048)
    export RUN_ID="${RUN_ID:-eval2048}"
    export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
    export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-2048}"
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    echo "Profiles: base10l zloss_low zloss_med twice_low zloss_twice eval2048" >&2
    exit 1
    ;;
esac

bash scripts/run_remote_experiment.sh "$@"
