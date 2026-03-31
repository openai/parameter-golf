#!/usr/bin/env bash
set -euo pipefail

# Strict lab protocol runner:
# - Never mutates pre-existing source files.
# - Copies train_gpt.py into a fresh per-run folder for each arm/seed.
# - Saves env snapshot + log + metrics + promotion summary.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GREENROD_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${GREENROD_DIR}/../.." && pwd)"
SOURCE_TRAIN="${SOURCE_TRAIN:-${GREENROD_DIR}/train_gpt.py}"
ARMS_TSV="${ARMS_TSV:-${SCRIPT_DIR}/concept_arms.tsv}"

if [[ ! -f "${SOURCE_TRAIN}" ]]; then
  echo "FATAL: source train script not found: ${SOURCE_TRAIN}" >&2
  exit 1
fi
if [[ ! -f "${ARMS_TSV}" ]]; then
  echo "FATAL: arms table not found: ${ARMS_TSV}" >&2
  exit 1
fi

export PATH="/home/frosty40/miniconda3/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

SEEDS_CSV="${SEEDS:-1337,1338}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
PROMOTE_DELTA="${PROMOTE_DELTA:-0.010}"

TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
COMPILE_ENABLED="${COMPILE_ENABLED:-0}"
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/ab1gpu_${RUN_TS}}"
METRICS_TSV="${RESULT_DIR}/metrics.tsv"
SUMMARY_TXT="${RESULT_DIR}/promotion_summary.txt"
mkdir -p "${RESULT_DIR}/runs"

echo -e "arm\tseed\tcap_step\tcap_val_bpb\trun_dir\tlog" > "${METRICS_TSV}"

echo "============================================"
echo "  1-GPU A/B Concept Runner (Strict Protocol)"
echo "  Source script: ${SOURCE_TRAIN}"
echo "  Arms table: ${ARMS_TSV}"
echo "  Seeds: ${SEEDS_CSV}"
echo "  Wallclock per run: ${MAX_WALLCLOCK_SECONDS}s"
echo "  Promotion threshold: ${PROMOTE_DELTA} BPB"
echo "  Results: ${RESULT_DIR}"
echo "============================================"

HAS_FLA=1
if ! python3 -c "from fla.layers.delta_net import DeltaNet" >/dev/null 2>&1; then
  HAS_FLA=0
  echo "WARNING: fla/DeltaNet not available; GDN arms will be skipped."
fi

extract_cap_metrics() {
  local log_path="$1"
  python3 - "$log_path" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"step:(\d+)/\d+ val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
if not matches:
    print("-\t-")
else:
    step, bpb = matches[-1]
    print(f"{step}\t{bpb}")
PY
}

run_arm_seed() {
  local arm="$1"
  local seed="$2"
  local gdn_enabled="$3"
  local gdn_num_layers="$4"
  local gdn_lr="$5"
  local xsa_last_n="$6"

  local run_dir="${RESULT_DIR}/runs/${arm}_s${seed}"
  local run_script="${run_dir}/train_gpt_copy.py"
  local run_log="${run_dir}/train.log"
  local env_file="${run_dir}/env_snapshot.txt"

  mkdir -p "${run_dir}"
  cp "${SOURCE_TRAIN}" "${run_script}"

  {
    echo "RUN_ID=ab1gpu_${arm}_s${seed}_${RUN_TS}"
    echo "SEED=${seed}"
    echo "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
    echo "VAL_LOSS_EVERY=${VAL_LOSS_EVERY}"
    echo "SKIP_FINAL_EVAL=${SKIP_FINAL_EVAL}"
    echo "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS}"
    echo "TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN}"
    echo "EVAL_SEQ_LEN=${EVAL_SEQ_LEN}"
    echo "COMPILE_ENABLED=${COMPILE_ENABLED}"
    echo "COMPILE_FULLGRAPH=${COMPILE_FULLGRAPH}"
    echo "COMPLEMENT_ALPHA=0"
    echo "NGRAM_EVAL_ORDER=0"
    echo "TRIGRAM=0"
    echo "NGRAM_ENTROPY_SHIFT=0"
    echo "GDN_ENABLED=${gdn_enabled}"
    echo "GDN_NUM_LAYERS=${gdn_num_layers}"
    echo "GDN_LR=${gdn_lr}"
    echo "XSA_LAST_N=${xsa_last_n}"
  } > "${env_file}"

  echo
  echo "==> arm=${arm} seed=${seed} (gdn=${gdn_enabled}/${gdn_num_layers}, lr=${gdn_lr}, xsa=${xsa_last_n})"

  if ! (
    export RUN_ID="ab1gpu_${arm}_s${seed}_${RUN_TS}"
    export SEED="${seed}"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}"
    export VAL_LOSS_EVERY="${VAL_LOSS_EVERY}"
    export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL}"
    export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}"
    export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN}"
    export EVAL_SEQ_LEN="${EVAL_SEQ_LEN}"
    export COMPILE_ENABLED="${COMPILE_ENABLED}"
    export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH}"
    export COMPLEMENT_ALPHA=0
    export NGRAM_EVAL_ORDER=0
    export TRIGRAM=0
    export NGRAM_ENTROPY_SHIFT=0
    export GDN_ENABLED="${gdn_enabled}"
    export GDN_NUM_LAYERS="${gdn_num_layers}"
    export GDN_LR="${gdn_lr}"
    export XSA_LAST_N="${xsa_last_n}"

    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      "${run_script}" \
      2>&1 | tee "${run_log}"
  ); then
    echo "WARNING: run failed for arm=${arm} seed=${seed}; continuing."
  fi

  local cap_step
  local cap_val_bpb
  read -r cap_step cap_val_bpb < <(extract_cap_metrics "${run_log}")
  echo -e "${arm}\t${seed}\t${cap_step}\t${cap_val_bpb}\t${run_dir}\t${run_log}" >> "${METRICS_TSV}"
}

while IFS=$'\t' read -r arm enabled gdn_enabled gdn_num_layers gdn_lr xsa_last_n notes; do
  [[ "${arm}" == "arm" ]] && continue
  [[ "${enabled}" != "1" ]] && continue

  if [[ "${gdn_enabled}" == "1" && "${HAS_FLA}" != "1" ]]; then
    echo "SKIP arm=${arm}: GDN requires fla, unavailable."
    continue
  fi

  for seed in "${SEEDS[@]}"; do
    run_arm_seed "${arm}" "${seed}" "${gdn_enabled}" "${gdn_num_layers}" "${gdn_lr}" "${xsa_last_n}"
  done
done < "${ARMS_TSV}"

echo
echo "==> Metrics"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${METRICS_TSV}"
else
  cat "${METRICS_TSV}"
fi

python3 - "${METRICS_TSV}" "${PROMOTE_DELTA}" > "${SUMMARY_TXT}" <<'PY'
import csv
import sys
from collections import defaultdict

metrics_tsv = sys.argv[1]
threshold = float(sys.argv[2])

rows = []
with open(metrics_tsv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

control = {}
arms = defaultdict(dict)
for r in rows:
    arm = r["arm"]
    seed = r["seed"]
    try:
        bpb = float(r["cap_val_bpb"])
    except ValueError:
        continue
    if arm == "control":
        control[seed] = bpb
    else:
        arms[arm][seed] = bpb

ranked = []
for arm, seed_map in arms.items():
    deltas = []
    for seed, bpb in seed_map.items():
        if seed in control:
            deltas.append(bpb - control[seed])
    if not deltas:
        continue
    mean_delta = sum(deltas) / len(deltas)
    strict_pass = all(d <= -threshold for d in deltas)
    ranked.append((mean_delta, strict_pass, arm, deltas))

ranked.sort(key=lambda x: x[0])

print("Promotion analysis (negative delta is better than control):")
print(f"Threshold per seed: delta <= {-threshold:.4f}")
if not ranked:
    print("No valid candidate data. PROMOTE: none")
    sys.exit(0)

for mean_delta, strict_pass, arm, deltas in ranked:
    ds = ", ".join(f"{d:+.4f}" for d in deltas)
    mark = "PASS" if strict_pass else "HOLD"
    print(f"{arm:10s} mean_delta={mean_delta:+.4f} per_seed=[{ds}] {mark}")

winner = next((x for x in ranked if x[1]), None)
if winner is None:
    print("PROMOTE: none")
else:
    print(f"PROMOTE: {winner[2]}")
PY

echo
echo "==> Promotion summary"
cat "${SUMMARY_TXT}"

echo
echo "Saved artifacts:"
echo "  ${METRICS_TSV}"
echo "  ${SUMMARY_TXT}"
echo "  ${RESULT_DIR}/runs/* (script copy + env snapshot + logs)"
