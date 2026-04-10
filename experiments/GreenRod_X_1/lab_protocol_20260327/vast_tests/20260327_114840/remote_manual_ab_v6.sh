#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="20260327_114840"
REPO="/workspace/parameter-golf-lab"
TEST_DIR="${REPO}/experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/${RUN_TAG}"
SOURCE_TRAIN="${REPO}/experiments/GreenRod_X_1/train_gpt.py"

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd "${REPO}"

python3 - <<'PY'
import torch
print("TORCH_READY", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
MAN_ROOT="${TEST_DIR}/manual_ab_v6_${STAMP}"
mkdir -p "${MAN_ROOT}/runs"
METRICS_TSV="${MAN_ROOT}/metrics.tsv"
SUMMARY_TXT="${MAN_ROOT}/promotion_summary.txt"
echo -e "arm\tseed\tcap_step\tcap_val_bpb\trun_dir\tlog" > "${METRICS_TSV}"

extract_cap_metrics() {
  local log_path="$1"
  python3 - "$log_path" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace')
matches = re.findall(r"step:(\d+)/\d+ val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
if not matches:
    print("-\t-")
else:
    s,b = matches[-1]
    print(f"{s}\t{b}")
PY
}

run_arm() {
  local arm="$1"
  local xsa="$2"
  local seed="1337"
  local run_dir="${MAN_ROOT}/runs/${arm}_s${seed}"
  local run_script="${run_dir}/train_gpt_copy.py"
  local run_log="${run_dir}/train.log"

  mkdir -p "${run_dir}"
  cp "${SOURCE_TRAIN}" "${run_script}"

  (
    export RUN_ID="manualab_${arm}_s${seed}_${STAMP}"
    export SEED="${seed}"
    export GDN_NUM_LAYERS=0
    export XSA_LAST_N="${xsa}"
    export NGRAM_EVAL_ORDER=0
    export TRIGRAM=0
    export NGRAM_ENTROPY_SHIFT=0
    export TRAIN_BATCH_TOKENS=16384
    export TRAIN_SEQ_LEN=1024
    export EVAL_SEQ_LEN=1024
    export ITERATIONS=80
    export WARMDOWN_ITERS=20
    export WARMUP_STEPS=2
    export MAX_WALLCLOCK_SECONDS=90
    export VAL_LOSS_EVERY=10
    export TRAIN_LOG_EVERY=10
    export SKIP_FINAL_EVAL=1
    export COMPILE_ENABLED=0
    export COMPILE_FULLGRAPH=0
    torchrun --standalone --nproc_per_node=1 "${run_script}" 2>&1 | tee "${run_log}"
  ) || true

  local cap_step cap_val_bpb
  read -r cap_step cap_val_bpb < <(extract_cap_metrics "${run_log}")
  echo -e "${arm}\t${seed}\t${cap_step}\t${cap_val_bpb}\t${run_dir}\t${run_log}" >> "${METRICS_TSV}"
}

run_arm control 11
run_arm a_xsa9 9

python3 - "${METRICS_TSV}" > "${SUMMARY_TXT}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8'), delimiter='\t'))
vals = {r['arm']: r for r in rows if r['cap_val_bpb'] not in ('-', '')}
print('Promotion analysis (manual_ab_v6)')
if 'control' not in vals or 'a_xsa9' not in vals:
    print('No valid candidate data. PROMOTE: none')
    sys.exit(0)
ctrl = float(vals['control']['cap_val_bpb'])
cand = float(vals['a_xsa9']['cap_val_bpb'])
delta = cand - ctrl
print(f"control={ctrl:.4f}")
print(f"a_xsa9={cand:.4f}")
print(f"delta(a_xsa9-control)={delta:+.4f}")
if delta <= -0.0100:
    print('PROMOTE: a_xsa9')
else:
    print('PROMOTE: none')
PY

cat "${METRICS_TSV}"
cat "${SUMMARY_TXT}"
echo "MAN_ROOT=${MAN_ROOT}"
