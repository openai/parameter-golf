#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

SKIP_QUANT=${SKIP_QUANT:-0}

run_one() {
  env "$@" bash records/track_non_record_16mb/2026-03-26_SemanticTube_11L_Study/run_semantic_tube_study.sh
}

run_one ROLE=control RUN_LABEL=t0_control_confirm TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=786432 SKIP_QUANT=${SKIP_QUANT}
run_one ROLE=tube    RUN_LABEL=t4_tube_confirm    TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=786432 SKIP_QUANT=${SKIP_QUANT}
run_one ROLE=control RUN_LABEL=s2_control_confirm TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 SKIP_QUANT=${SKIP_QUANT}
run_one ROLE=tube    RUN_LABEL=s3_tube_confirm    TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 SKIP_QUANT=${SKIP_QUANT}
