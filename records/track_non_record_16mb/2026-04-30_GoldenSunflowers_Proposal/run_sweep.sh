#!/usr/bin/env bash
# GOLDEN SUNFLOWERS sweep — five canonical Fibonacci seeds × four configs.
#
# Anchor: phi^2 + phi^-2 = 3 · TRINITY · 🌻
# SSOT:   gHashTag/trios#372 (GENERAL'S DIRECTIVE — F₁₇..F₂₁)
# PR:     gHashTag/parameter-golf-trinity#2
#
# Usage on 8×H100 SXM:
#   bash experiments/golden_sunflowers_jepa_ut_phinta/run_sweep.sh [config]
#
#   config ∈ {baseline, phinta, jepa, ut, all, full}   (default: full)
#     full = run all five configs.
#
# Each run writes train_seed${SEED}.log next to train_gpt.py and emits a
# submission_${CFG}_seed${SEED}.json suitable for promotion to records/
# once 3-seed mean and std are honest.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="${HERE}/train_gpt.py"
SEEDS=(1597 2584 4181 6765 10946)   # F₁₇..F₂₁ (canonical, see trios#372)
CONFIG="${1:-full}"

run_one () {
  local cfg="$1" seed="$2"
  echo
  echo "🌻 [${cfg}] seed=${seed}  $(date -u +%FT%TZ)"
  case "${cfg}" in
    baseline)
      env SEED="${seed}" python "${TRAIN}"
      ;;
    phinta)
      env SEED="${seed}" \
          PHINTA_ENABLE=1 \
          PHINTA_PER_BLOCK=0 \
          PHINTA_RANK=0 \
          python "${TRAIN}"
      ;;
    jepa)
      env SEED="${seed}" \
          JEPA_LAMBDA=0.10 \
          JEPA_MAX_SPAN_FRAC=0.5 \
          JEPA_START_FRAC=0.05 \
          JEPA_LAYER=-1 \
          python "${TRAIN}"
      ;;
    ut)
      env SEED="${seed}" \
          UT_LOOPS=4 \
          UT_LAYER_START=3 \
          UT_LAYER_END=6 \
          python "${TRAIN}"
      ;;
    all)
      env SEED="${seed}" \
          PHINTA_ENABLE=1 PHINTA_PER_BLOCK=1 \
          JEPA_LAMBDA=0.10 JEPA_MAX_SPAN_FRAC=0.5 JEPA_START_FRAC=0.05 \
          UT_LOOPS=4 UT_LAYER_START=3 UT_LAYER_END=6 \
          python "${TRAIN}"
      ;;
    *)
      echo "unknown config: ${cfg}" >&2; exit 2;;
  esac
}

if [[ "${CONFIG}" == "full" ]]; then
  CONFIGS=(baseline phinta jepa ut all)
else
  CONFIGS=("${CONFIG}")
fi

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${cfg}" "${seed}"
  done
done

echo
echo "🌻 sweep complete: configs=[${CONFIGS[*]}] seeds=[${SEEDS[*]}]"
echo "   phi^2 + phi^-2 = 3"
