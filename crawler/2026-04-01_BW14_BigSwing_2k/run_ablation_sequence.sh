#!/bin/bash
set -euo pipefail
# ================================================================
#  BW14_BigSwing_2k — Unified Ablation Sequence
#
#  Goal: hunt for another ~0.01-class drop via architecture phase shifts,
#  not micro quant/polish tweaks.
#
#  WINDOW-only arms (all require retrain):
#    - depth phase shift (5F -> 6F)
#    - crawler choke routing phase shift (flat/residual variants)
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-01_BW14_BigSwing_2k/run_ablation_sequence.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_PY="${REPO_ROOT}/crawler/2026-04-01_BW12_Interaction_2k/train_gpt_ablate.py"
RUN_TAG="BW14BS"
TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary_s${SEED}_${TS}.tsv"

TORCHRUN=$(command -v torchrun 2>/dev/null || echo "python3 -m torch.distributed.run")

# Shared base: BW13 control (tap-off Nightcrawler), 2k signal gate
BASE_ENV=(
    SEED="${SEED}"
    ITERATIONS=2000
    MAX_WALLCLOCK_SECONDS=3600
    WARMDOWN_ITERS=2000
    COMPLEMENT_ALPHA=0
    XSA_LAST_N=11
    BIGRAM_VOCAB_SIZE=2048
    ROPE_DIMS=16
    SWA_EVERY=50
    MTP_NUM_HEADS=0
    LATE_QAT_THRESHOLD=0
    MATRIX_LR=0.03
    TORCHDYNAMO_OPTIMIZE_DDP=0
    COMPILE_FULLGRAPH=1
    NGRAM_EVAL_ORDER=0
    MODEL_DIM=512
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=5
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=3
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8=1
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_MLP_CHOKE_SHAPE=flat
    CRAWLER_MLP_CHOKE_GROUPS=8
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=1
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    NPROC_PER_NODE="${NPROC}"
)

CONTROL_INT6=""
CONTROL_STEP_MS=""

{
    echo -e "lane\tarm\tdesc\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tdelta_vs_control\tlog"
} > "${SUMMARY}"

calc_delta() {
    local value="$1"
    if [[ -z "${CONTROL_INT6}" || -z "${value}" || "${value}" == "?" ]]; then
        echo "?"
        return
    fi
    python3 - <<PY
c = float("${CONTROL_INT6}")
v = float("${value}")
d = v - c
sign = "+" if d >= 0 else ""
print(f"{sign}{d:.6f}")
PY
}

extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

run_arm() {
    local arm="$1"; shift
    local desc="$1"; shift
    local extra_env=("$@")

    local log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  WINDOW ${arm}: ${desc}"
    echo "  log: ${log}"
    echo "----------------------------------------------------------------"

    env "${BASE_ENV[@]}" "${extra_env[@]}" \
      ${TORCHRUN} --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
      2>&1 | tee "${log}"

    local params raw int6 step_ms bytes delta
    params=$(extract_metric 'model_params:\K[0-9]+' "${log}")
    raw=$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    int6=$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    step_ms=$(extract_metric 'step_avg:\K[0-9.]+' "${log}")
    bytes=$(extract_metric 'Total submission size int6\+zstd: \K[0-9]+' "${log}")

    if [[ -z "${params}" ]]; then params="?"; fi
    if [[ -z "${raw}" ]]; then raw="?"; fi
    if [[ -z "${int6}" ]]; then int6="?"; fi
    if [[ -z "${step_ms}" ]]; then step_ms="?"; fi
    if [[ -z "${bytes}" ]]; then bytes="?"; fi

    if [[ "${arm}" == "${RUN_TAG}-00" ]]; then
        CONTROL_INT6="${int6}"
        CONTROL_STEP_MS="${step_ms}"
    fi

    delta=$(calc_delta "${int6}")
    echo -e "WINDOW\t${arm}\t${desc}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${delta}\t${log}" >> "${SUMMARY}"

    echo "  ${arm}: params=${params} raw=${raw} int6_sw=${int6} step_ms=${step_ms} bytes=${bytes} delta_vs_ctrl=${delta}"
}

# ----------------------------------------------------------------
# WINDOW: high-upside architecture phase-shift tests
# ----------------------------------------------------------------
run_arm "${RUN_TAG}-00" "control (tap-off Nightcrawler, naive int6)"

run_arm "${RUN_TAG}-01" "depth phase shift: NUM_FLAT_LAYERS=6" \
    NUM_FLAT_LAYERS=6

run_arm "${RUN_TAG}-02" "crawler choke flat-128" \
    CRAWLER_MLP_CHOKE_DIM=128 \
    CRAWLER_MLP_CHOKE_SHAPE=flat

run_arm "${RUN_TAG}-03" "crawler choke flat-512" \
    CRAWLER_MLP_CHOKE_DIM=512 \
    CRAWLER_MLP_CHOKE_SHAPE=flat

run_arm "${RUN_TAG}-04" "crawler choke residual-128" \
    CRAWLER_MLP_CHOKE_DIM=128 \
    CRAWLER_MLP_CHOKE_SHAPE=residual

cat <<TXT

================================================================
BW14 big-swing sequence complete.
summary: ${SUMMARY}

Promotion policy (full 600s 8xH100):
  1) Big-swing promote: delta_vs_control <= -0.0060
  2) Secondary promote: -0.0060 < delta_vs_control <= -0.0030
  3) Guardrails: bytes <= 16MB and step_ms not catastrophically slower

Interpretation:
  - This leg is intentionally high-variance. Positive deltas are expected for some arms.
  - Goal is to locate phase-shift behavior, not polish an already-known quant trick.
================================================================
TXT

column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
