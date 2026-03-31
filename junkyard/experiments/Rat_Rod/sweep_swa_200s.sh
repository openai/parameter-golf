#!/usr/bin/env bash
set -euo pipefail

# Rat Rod zero-overhead sweep: SWA_EVERY
# Default values compare baseline 50 vs candidate 100.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEEDS="${SEEDS:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WALLCLOCK_SECONDS="${WALLCLOCK_SECONDS:-200}"
SWA_VALUES="${SWA_VALUES:-50 100}"

# Runtime controls (allow cheap screening profile on fragile pods)
COMPILE_ENABLED="${COMPILE_ENABLED:-1}"
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-1}"
TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-0}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-9}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-results/ratrod_sweeps}"
mkdir -p "${RESULT_DIR}" logs
SUMMARY_TSV="${RESULT_DIR}/swa_200s_${RUN_TS}.tsv"

echo -e "sweep\tseed\tvalue\tcap_step\tcap_val_bpb\tdiag_bpb\tsliding_bpb\tngram9_bpb\tpeak_alloc_mib\tlog" > "${SUMMARY_TSV}"

echo "============================================"
echo "  Rat Rod Sweep: SWA_EVERY"
echo "  Seeds: ${SEEDS}"
echo "  Values: ${SWA_VALUES}"
echo "  Wallclock: ${WALLCLOCK_SECONDS}s"
echo "  NPROC: ${NPROC_PER_NODE}"
echo "============================================"

for seed in ${SEEDS//,/ }; do
  for swa in ${SWA_VALUES}; do
    LOG_PATH="logs/sweep_swa_${swa}_s${seed}_${RUN_TS}.log"
    echo
    echo "==> seed=${seed} SWA_EVERY=${swa}"
    SEED="${seed}" \
    MAX_WALLCLOCK_SECONDS="${WALLCLOCK_SECONDS}" \
    WARMDOWN_ITERS=3500 \
    COMPLEMENT_ALPHA=0 \
    XSA_LAST_N=11 \
    BIGRAM_VOCAB_SIZE=2048 \
    ROPE_DIMS=16 \
    SWA_EVERY="${swa}" \
    MTP_NUM_HEADS=0 \
    TRIGRAM=0 \
    LATE_QAT_THRESHOLD=0 \
    NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER}" \
    NGRAM_EVAL_MIN_ORDER=2 \
    NGRAM_EVAL_ADAPTIVE=1 \
    NGRAM_EVAL_ALPHA=0.30 \
    NGRAM_EVAL_ALPHA_MIN=0.05 \
    NGRAM_EVAL_ALPHA_MAX=0.60 \
    NGRAM_EVAL_ENTROPY_CENTER=3.0 \
    NGRAM_EVAL_ENTROPY_SCALE=2.0 \
    NGRAM_EVAL_MIN_COUNT=2 \
    NGRAM_EVAL_BUCKETS=8388608 \
    NGRAM_EVAL_MAX_SECONDS=0 \
    CUBRIC_CADENCE=0 \
    NGRAM_ENTROPY_SHIFT=1 \
    NGRAM_ORDER_MULTS="0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0" \
    COMPILE_ENABLED="${COMPILE_ENABLED}" \
    COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH}" \
    TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE}" \
    SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL}" \
    TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      "${REPO_ROOT}/experiments/Rat_Rod/green/train_gpt.py" \
      2>&1 | tee "${LOG_PATH}"

    python3 "${REPO_ROOT}/experiments/Rat_Rod/parse_ratrod_log.py" \
      --log "${LOG_PATH}" \
      --sweep swa \
      --seed "${seed}" \
      --value "${swa}" \
      --ngram-order "${NGRAM_EVAL_ORDER}" >> "${SUMMARY_TSV}"
  done
done

echo
echo "Summary TSV: ${SUMMARY_TSV}"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${SUMMARY_TSV}"
else
  cat "${SUMMARY_TSV}"
fi
