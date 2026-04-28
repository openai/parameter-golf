#!/bin/bash
set -euo pipefail
# ================================================================
# BWX_Latest_2k
# Primary contender sequence from BW12..BW16 findings:
#   1) WINDOW (full train): big swings first, small depth sanity last
#   2) POST_WINDOW (no retrain): quant bake-off on best WINDOW checkpoint
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_PY="${TRAIN_PY:-${REPO_ROOT}/crawler/2026-04-01_BW12_Interaction_2k/train_gpt_ablate.py}"
RUN_TAG="BWXLT"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOOP_AWARE_GPTQ="${RUN_LOOP_AWARE_GPTQ:-0}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary_s${SEED}_${TS}.tsv"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

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
    NUM_FLAT_LAYERS=8
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
    GPTQ_CAL_SAMPLES=128
    GPTQ_CAL_SEQ_LEN=2048
    NPROC_PER_NODE="${NPROC}"
)

CONTROL_ARM="${RUN_TAG}-00"
CONTROL_INT6=""
CONTROL_CKPT=""
BEST_WINDOW_ARM=""
BEST_WINDOW_DEPTH=""
BEST_WINDOW_INT6=""
BEST_WINDOW_CKPT=""
BEST_WINDOW_DESC=""

{
    echo -e "lane\tarm\tdesc\tnum_flat_layers\tsource_ckpt\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tgptq_layers\tgptq_cal_sec\tdelta_vs_control\tlog"
} > "${SUMMARY}"

extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

is_numeric() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

calc_delta() {
    local value="$1"
    if ! is_numeric "${CONTROL_INT6}" || ! is_numeric "${value}"; then
        echo "?"
        return
    fi
    python3 - "${CONTROL_INT6}" "${value}" <<'PY'
import sys
c = float(sys.argv[1])
v = float(sys.argv[2])
d = v - c
sign = "+" if d >= 0 else ""
print(f"{sign}{d:.6f}")
PY
}

is_better_window() {
    local candidate="$1"
    local incumbent="$2"
    if ! is_numeric "${candidate}"; then
        return 1
    fi
    if ! is_numeric "${incumbent}"; then
        return 0
    fi
    python3 - "${candidate}" "${incumbent}" <<'PY'
import sys
cand = float(sys.argv[1])
best = float(sys.argv[2])
sys.exit(0 if cand < best else 1)
PY
}

run_arm() {
    local lane="$1"; shift
    local arm="$1"; shift
    local desc="$1"; shift
    local depth="$1"; shift
    local source_ckpt="$1"; shift
    local extra_env=("$@")

    local log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  ${lane} ${arm}: ${desc}"
    echo "  log: ${log}"
    echo "----------------------------------------------------------------"

    env "${BASE_ENV[@]}" NUM_FLAT_LAYERS="${depth}" "${extra_env[@]}" \
      "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
      2>&1 | tee "${log}"

    local params raw int6 step_ms bytes gptq_layers gptq_cal_sec delta
    params=$(extract_metric 'model_params:\K[0-9]+' "${log}")
    raw=$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    int6=$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    step_ms=$(extract_metric 'step_avg:\K[0-9.]+' "${log}")
    bytes=$(extract_metric 'Total submission size int6\+(?:zstd|zlib): \K[0-9]+' "${log}")
    gptq_layers=$(extract_metric 'gptq_quantize: \K[0-9]+' "${log}")
    gptq_cal_sec=$(extract_metric 'gptq:(?:loop-aware )?calibrated [0-9]+ layers in \K[0-9.]+' "${log}")

    if [[ -z "${params}" ]]; then params="?"; fi
    if [[ -z "${raw}" ]]; then raw="?"; fi
    if [[ -z "${int6}" ]]; then int6="?"; fi
    if [[ -z "${step_ms}" ]]; then step_ms="?"; fi
    if [[ -z "${bytes}" ]]; then bytes="?"; fi
    if [[ -z "${gptq_layers}" ]]; then gptq_layers="0"; fi
    if [[ -z "${gptq_cal_sec}" ]]; then gptq_cal_sec="-"; fi

    if [[ "${arm}" == "${CONTROL_ARM}" ]]; then
        CONTROL_INT6="${int6}"
    fi

    delta=$(calc_delta "${int6}")
    echo -e "${lane}\t${arm}\t${desc}\t${depth}\t${source_ckpt}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${gptq_layers}\t${gptq_cal_sec}\t${delta}\t${log}" >> "${SUMMARY}"

    if [[ "${lane}" == "WINDOW" ]]; then
        local arm_ckpt="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.final_model.pt"
        if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
            cp -f "${REPO_ROOT}/final_model.pt" "${arm_ckpt}"
            if [[ "${arm}" == "${CONTROL_ARM}" ]]; then
                CONTROL_CKPT="${arm_ckpt}"
            fi
            if is_better_window "${int6}" "${BEST_WINDOW_INT6}"; then
                BEST_WINDOW_ARM="${arm}"
                BEST_WINDOW_DEPTH="${depth}"
                BEST_WINDOW_INT6="${int6}"
                BEST_WINDOW_CKPT="${arm_ckpt}"
                BEST_WINDOW_DESC="${desc}"
            fi
        else
            echo "  WARNING: final_model.pt missing after WINDOW arm ${arm}" >&2
        fi
    fi

    echo "  ${arm}: raw=${raw} int6_sw=${int6} step_ms=${step_ms} bytes=${bytes} gptq_layers=${gptq_layers} gptq_cal_sec=${gptq_cal_sec} delta_vs_ctrl=${delta}"
}

run_window_arm() {
    local arm="$1"; shift
    local desc="$1"; shift
    local depth="$1"; shift
    run_arm "WINDOW" "${arm}" "${desc}" "${depth}" "-" "$@"
}

run_post_window_arm() {
    local arm="$1"; shift
    local desc="$1"; shift
    if [[ -z "${BEST_WINDOW_CKPT}" || ! -f "${BEST_WINDOW_CKPT}" ]]; then
        echo "ERROR: missing best WINDOW checkpoint: ${BEST_WINDOW_CKPT:-(unset)}" >&2
        exit 1
    fi

    run_arm "POST_WINDOW" "${arm}" "${desc}" "${BEST_WINDOW_DEPTH}" "${BEST_WINDOW_CKPT}" \
        SKIP_TRAIN=1 \
        INIT_MODEL_PATH="${BEST_WINDOW_CKPT}" \
        "$@"
}

# ----------------------------------------------------------------
# 1) WINDOW arms first: big swings first, small sanity last.
#    Control is the viable contender: tap-off + deeper floor + no anchor.
# ----------------------------------------------------------------
run_window_arm "${RUN_TAG}-00" "control contender (tap-off, no anchor, NUM_FLAT_LAYERS=8, naive int6)" 8
run_window_arm "${RUN_TAG}-06" "big swing retest (tap-off, no anchor, NUM_FLAT_LAYERS=6)" 6
run_window_arm "${RUN_TAG}-07" "depth sanity below contender (NUM_FLAT_LAYERS=7)" 7
run_window_arm "${RUN_TAG}-09" "depth sanity above contender, size-risk check (NUM_FLAT_LAYERS=9)" 9

if [[ -z "${BEST_WINDOW_ARM}" ]]; then
    echo "ERROR: no viable WINDOW arm found for post-window quant stage" >&2
    exit 1
fi

echo ""
echo "Best WINDOW checkpoint selected for post-window quant:"
echo "  arm=${BEST_WINDOW_ARM} depth=${BEST_WINDOW_DEPTH} int6_sw_bpb=${BEST_WINDOW_INT6}"
echo "  ckpt=${BEST_WINDOW_CKPT}"

# ----------------------------------------------------------------
# 2) POST_WINDOW quant bake-off on best WINDOW checkpoint.
# ----------------------------------------------------------------
run_post_window_arm "${RUN_TAG}-Q0" "naive int6 on frozen best WINDOW checkpoint" \
    SKIP_GPTQ=1 \
    LOOP_AWARE_GPTQ=0

run_post_window_arm "${RUN_TAG}-Q1" "standard GPTQ (128x2048) on frozen best WINDOW checkpoint" \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=0 \
    GPTQ_CAL_SAMPLES=128 \
    GPTQ_CAL_SEQ_LEN=2048

run_post_window_arm "${RUN_TAG}-Q1L" "standard GPTQ-lite (64x1024) on frozen best WINDOW checkpoint" \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=0 \
    GPTQ_CAL_SAMPLES=64 \
    GPTQ_CAL_SEQ_LEN=1024

if [[ "${RUN_LOOP_AWARE_GPTQ}" == "1" ]]; then
    run_post_window_arm "${RUN_TAG}-Q2" "loop-aware GPTQ (optional) on frozen best WINDOW checkpoint" \
        SKIP_GPTQ=0 \
        LOOP_AWARE_GPTQ=1 \
        GPTQ_CAL_SAMPLES=128 \
        GPTQ_CAL_SEQ_LEN=2048
fi

cat <<TXT

================================================================
BWX latest contender sequence complete.
summary: ${SUMMARY}
control arm: ${CONTROL_ARM} (delta baseline for all lanes)
best WINDOW arm for quant: ${BEST_WINDOW_ARM} (depth=${BEST_WINDOW_DEPTH}, int6_sw_bpb=${BEST_WINDOW_INT6})
loop-aware GPTQ arm: $( [[ "${RUN_LOOP_AWARE_GPTQ}" == "1" ]] && echo "enabled" || echo "disabled (set RUN_LOOP_AWARE_GPTQ=1 to include)" )
================================================================
TXT

column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
