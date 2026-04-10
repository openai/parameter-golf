#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-proxy}" # proxy | full
SEEDS_CSV="${SEEDS:-444}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

CONTROL_SCRIPT="${CONTROL_SCRIPT:-${REPO_ROOT}/records/track_10min_16mb/2026-03-29_Bandit_ClownCar_X_CubricNgram9_8xH100/train_gpt.py}"
CODEX_SCRIPT="${CODEX_SCRIPT:-${SCRIPT_DIR}/train_gpt_BANDIT_CODEX.py}"
EXPECTED_CONTROL_SHA="${EXPECTED_CONTROL_SHA:-b3fcfee4bebe4572d8e181dc20cc526737e40c08fcf28db56a1076432440be22}"

if [[ ! -f "${CONTROL_SCRIPT}" ]]; then
  echo "FATAL: missing control script: ${CONTROL_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${CODEX_SCRIPT}" ]]; then
  echo "FATAL: missing BANDIT CODEX script: ${CODEX_SCRIPT}" >&2
  exit 1
fi

control_sha="$(sha256sum "${CONTROL_SCRIPT}" | awk '{print $1}')"
if [[ "${control_sha}" != "${EXPECTED_CONTROL_SHA}" ]]; then
  echo "FATAL: control SHA mismatch." >&2
  echo "  expected: ${EXPECTED_CONTROL_SHA}" >&2
  echo "  actual:   ${control_sha}" >&2
  echo "Refusing to run A/B on an unverified control baseline." >&2
  exit 2
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/ab_results_${MODE}_${RUN_TS}}"
mkdir -p "${RESULT_DIR}"
METRICS_TSV="${RESULT_DIR}/metrics.tsv"
SUMMARY_TXT="${RESULT_DIR}/summary.txt"

echo -e "mode\tarm\tseed\tscript\tmetric_name\tmetric\tlog_path" > "${METRICS_TSV}"

BASELINE_TOL="${BASELINE_TOL:-0.0015}"
NPROC_PROXY="${NPROC_PROXY:-1}"
NPROC_FULL="${NPROC_FULL:-8}"

parse_metric() {
  local mode="$1"
  local log_path="$2"
  python3 - "$mode" "$log_path" <<'PY'
import re
import sys
from pathlib import Path

mode = sys.argv[1]
log_path = Path(sys.argv[2])
text = log_path.read_text(encoding="utf-8", errors="replace")

def last(pattern: str):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None

if mode == "proxy":
    step = last(r"step:(\d+)/\d+\s+val_loss:[0-9.]+\s+val_bpb:([0-9.]+)")
    if step:
        print(f"proxy_step_val_bpb\t{step[1]}")
        sys.exit(0)
    sw = last(r"final_sliding_window_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)")
    if sw:
        print(f"final_sliding_window_exact\t{sw}")
        sys.exit(0)
    print("missing\tNaN")
    sys.exit(0)

ng = last(r"final_int6_sliding_window_ngram9_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)")
if ng:
    print(f"final_ngram9_exact\t{ng}")
    sys.exit(0)
sw = last(r"final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)")
if sw:
    print(f"final_int6_sliding_window_exact\t{sw}")
    sys.exit(0)
step = last(r"step:(\d+)/\d+\s+val_loss:[0-9.]+\s+val_bpb:([0-9.]+)")
if step:
    print(f"fallback_step_val_bpb\t{step[1]}")
    sys.exit(0)
print("missing\tNaN")
PY
}

expected_seed_metric() {
  local seed="$1"
  case "${seed}" in
    4)   printf "0.49638543" ;;
    300) printf "0.49606916" ;;
    444) printf "0.49571114" ;;
    *)   printf "" ;;
  esac
}

verify_full_baseline() {
  local seed="$1"
  local observed="$2"
  local expected
  expected="$(expected_seed_metric "${seed}")"
  if [[ -z "${expected}" ]]; then
    echo "FATAL: no recorded reference metric for seed ${seed}; cannot verify baseline." >&2
    exit 3
  fi
  python3 - "${observed}" "${expected}" "${BASELINE_TOL}" <<'PY'
import math
import sys
obs = float(sys.argv[1])
exp = float(sys.argv[2])
tol = float(sys.argv[3])
delta = abs(obs - exp)
if delta > tol:
    print(f"FAIL baseline verify: observed={obs:.8f} expected={exp:.8f} abs_delta={delta:.8f} tol={tol:.8f}")
    raise SystemExit(1)
print(f"PASS baseline verify: observed={obs:.8f} expected={exp:.8f} abs_delta={delta:.8f} tol={tol:.8f}")
PY
}

run_arm() {
  local arm="$1"
  local seed="$2"
  local script_path
  local turbomuon="0"
  local engramlite="0"
  local nproc
  local log_path="${RESULT_DIR}/${arm}_s${seed}.log"

  if [[ "${MODE}" == "proxy" ]]; then
    nproc="${NPROC_PROXY}"
  else
    nproc="${NPROC_FULL}"
  fi

  case "${arm}" in
    control)
      script_path="${CONTROL_SCRIPT}"
      ;;
    turbomuon)
      script_path="${CODEX_SCRIPT}"
      turbomuon="1"
      ;;
    engramlite)
      script_path="${CODEX_SCRIPT}"
      engramlite="1"
      ;;
    both)
      script_path="${CODEX_SCRIPT}"
      turbomuon="1"
      engramlite="1"
      ;;
    *)
      echo "FATAL: unknown arm ${arm}" >&2
      exit 4
      ;;
  esac

  echo
  echo "==> mode=${MODE} arm=${arm} seed=${seed} nproc=${nproc}"
  echo "    script=${script_path}"

  if [[ "${MODE}" == "proxy" ]]; then
    env \
      SEED="${seed}" \
      RUN_ID="ab_${MODE}_${arm}_s${seed}_${RUN_TS}" \
      TURBOMUON="${turbomuon}" \
      ENGRAMLITE="${engramlite}" \
      MAX_WALLCLOCK_SECONDS=0 \
      ITERATIONS="${PROXY_ITERATIONS:-1600}" \
      WARMDOWN_ITERS="${PROXY_WARMDOWN_ITERS:-400}" \
      WARMUP_STEPS="${PROXY_WARMUP_STEPS:-20}" \
      TRAIN_BATCH_TOKENS="${PROXY_TRAIN_BATCH_TOKENS:-131072}" \
      VAL_BATCH_SIZE="${PROXY_VAL_BATCH_SIZE:-131072}" \
      VAL_LOSS_EVERY="${PROXY_VAL_LOSS_EVERY:-200}" \
      TRAIN_LOG_EVERY="${PROXY_TRAIN_LOG_EVERY:-100}" \
      SKIP_FINAL_EVAL=1 \
      NGRAM_EVAL_ORDER=0 \
      COMPILE_ENABLED="${PROXY_COMPILE_ENABLED:-0}" \
      COMPILE_FULLGRAPH="${PROXY_COMPILE_FULLGRAPH:-0}" \
      COMPLEMENT_ALPHA=0.5 \
      XSA_LAST_N=11 \
      BIGRAM_VOCAB_SIZE=2048 \
      ROPE_DIMS=16 \
      MATRIX_LR=0.03 \
      USE_CRAWLER=1 \
      NUM_FLAT_LAYERS=4 \
      NUM_CRAWLER_LAYERS=1 \
      CRAWLER_LOOPS=4 \
      INST_DIM=32 \
      CRAWLER_QUANT_INT8=1 \
      DELTA_NET_HEADS=0 \
      EMA_START_STEP=4400 \
      EMA_DECAY=0.99 \
      LOOP_AWARE_GPTQ=1 \
      torchrun --standalone --nproc_per_node="${nproc}" "${script_path}" \
      2>&1 | tee "${log_path}"
  else
    env \
      SEED="${seed}" \
      RUN_ID="ab_${MODE}_${arm}_s${seed}_${RUN_TS}" \
      TURBOMUON="${turbomuon}" \
      ENGRAMLITE="${engramlite}" \
      MAX_WALLCLOCK_SECONDS="${FULL_MAX_WALLCLOCK_SECONDS:-600}" \
      WARMDOWN_ITERS="${FULL_WARMDOWN_ITERS:-2000}" \
      COMPLEMENT_ALPHA=0.5 \
      XSA_LAST_N=11 \
      BIGRAM_VOCAB_SIZE=2048 \
      ROPE_DIMS=16 \
      SWA_EVERY=50 \
      MTP_NUM_HEADS=0 \
      LATE_QAT_THRESHOLD=0 \
      MATRIX_LR=0.03 \
      TORCHDYNAMO_OPTIMIZE_DDP=0 \
      COMPILE_FULLGRAPH=0 \
      USE_CRAWLER=1 \
      NUM_FLAT_LAYERS=4 \
      NUM_CRAWLER_LAYERS=1 \
      CRAWLER_LOOPS=4 \
      INST_DIM=32 \
      CRAWLER_QUANT_INT8=1 \
      DELTA_NET_HEADS=0 \
      EMA_START_STEP=4400 \
      EMA_DECAY=0.99 \
      LOOP_AWARE_GPTQ=1 \
      NGRAM_EVAL_ORDER=9 \
      NGRAM_EVAL_MIN_ORDER=2 \
      NGRAM_EVAL_ADAPTIVE=1 \
      NGRAM_EVAL_ALPHA=0.30 \
      NGRAM_EVAL_ALPHA_MIN=0.20 \
      NGRAM_EVAL_ALPHA_MAX=0.75 \
      NGRAM_EVAL_ENTROPY_CENTER=3.0 \
      NGRAM_EVAL_ENTROPY_SCALE=2.0 \
      NGRAM_EVAL_MIN_COUNT=2 \
      NGRAM_EVAL_BUCKETS=8388608 \
      CUBRIC_CADENCE=32 \
      torchrun --standalone --nproc_per_node="${nproc}" "${script_path}" \
      2>&1 | tee "${log_path}"
  fi

  local metric_name metric
  read -r metric_name metric < <(parse_metric "${MODE}" "${log_path}")
  echo -e "${MODE}\t${arm}\t${seed}\t${script_path}\t${metric_name}\t${metric}\t${log_path}" >> "${METRICS_TSV}"
  echo "    metric=${metric_name}:${metric}"

  if [[ "${MODE}" == "full" && "${arm}" == "control" ]]; then
    verify_full_baseline "${seed}" "${metric}"
  fi
}

echo "============================================"
echo "BANDIT CODEX A/B"
echo "mode=${MODE}"
echo "seeds=${SEEDS_CSV}"
echo "control=${CONTROL_SCRIPT}"
echo "codex=${CODEX_SCRIPT}"
echo "control_sha=${control_sha}"
echo "results=${RESULT_DIR}"
echo "============================================"

for seed in "${SEEDS[@]}"; do
  run_arm "control" "${seed}"
  run_arm "turbomuon" "${seed}"
  run_arm "engramlite" "${seed}"
  run_arm "both" "${seed}"
done

python3 - "${METRICS_TSV}" "${SUMMARY_TXT}" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

metrics_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

rows = []
with metrics_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

control = {}
arms = defaultdict(dict)
for r in rows:
    arm = r["arm"]
    seed = r["seed"]
    try:
      metric = float(r["metric"])
    except ValueError:
      continue
    if not math.isfinite(metric):
      continue
    if arm == "control":
      control[seed] = metric
    else:
      arms[arm][seed] = metric

lines = []
lines.append("Paired deltas vs control (negative is better):")
for arm in ("turbomuon", "engramlite", "both"):
    deltas = []
    for seed, metric in sorted(arms.get(arm, {}).items(), key=lambda x: int(x[0])):
        if seed in control:
            deltas.append((seed, metric - control[seed]))
    if not deltas:
        lines.append(f"{arm:10s} no paired data")
        continue
    mean_delta = sum(d for _, d in deltas) / len(deltas)
    per_seed = ", ".join(f"s{seed}:{delta:+.6f}" for seed, delta in deltas)
    lines.append(f"{arm:10s} mean_delta={mean_delta:+.6f}  {per_seed}")

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_path.read_text(encoding="utf-8"), end="")
PY

if [[ "${MODE}" == "proxy" ]]; then
  echo
  echo "NOTE: proxy deltas are screening-only and must not be promoted directly."
  echo "Use MODE=full to enforce baseline reproduction before any expensive promotion run."
fi

echo
echo "Saved:"
echo "  ${METRICS_TSV}"
echo "  ${SUMMARY_TXT}"
