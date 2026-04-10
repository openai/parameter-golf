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
ROOT="${TEST_DIR}/proxy_ab_v7_${STAMP}"
mkdir -p "${ROOT}/runs"
METRICS="${ROOT}/metrics.tsv"
SUMMARY="${ROOT}/summary.txt"
echo -e "arm\tseed\tlast_step\tlast_train_loss\trun_dir\tlog" > "${METRICS}"

run_arm() {
  local arm="$1"
  local xsa="$2"
  local seed="1337"
  local rdir="${ROOT}/runs/${arm}_s${seed}"
  local rpy="${rdir}/train_gpt_copy.py"
  local rlog="${rdir}/train.log"
  mkdir -p "${rdir}"
  cp "${SOURCE_TRAIN}" "${rpy}"

  (
    export RUN_ID="proxyab_${arm}_s${seed}_${STAMP}"
    export SEED="${seed}"
    export GDN_NUM_LAYERS=0
    export XSA_LAST_N="${xsa}"
    export NGRAM_EVAL_ORDER=0
    export TRIGRAM=0
    export NGRAM_ENTROPY_SHIFT=0
    export TRAIN_BATCH_TOKENS=16384
    export TRAIN_SEQ_LEN=1024
    export EVAL_SEQ_LEN=1024
    export ITERATIONS=40
    export WARMDOWN_ITERS=10
    export WARMUP_STEPS=2
    export MAX_WALLCLOCK_SECONDS=90
    export VAL_LOSS_EVERY=1000000
    export TRAIN_LOG_EVERY=10
    export SKIP_FINAL_EVAL=1
    export COMPILE_ENABLED=0
    export COMPILE_FULLGRAPH=0
    torchrun --standalone --nproc_per_node=1 "${rpy}" 2>&1 | tee "${rlog}"
  ) || true

  python3 - "${rlog}" "${arm}" "${seed}" "${rdir}" "${rlog}" >> "${METRICS}" <<'PY'
import re, sys
from pathlib import Path
log = Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace')
arm, seed, rdir, rlog = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
m = re.findall(r"step:(\d+)/(\d+) train_loss:([0-9.]+)", log)
if not m:
    print(f"{arm}\t{seed}\t-\t-\t{rdir}\t{rlog}")
else:
    step, _, loss = m[-1]
    print(f"{arm}\t{seed}\t{step}\t{loss}\t{rdir}\t{rlog}")
PY
}

run_arm control 11
run_arm a_xsa9 9

python3 - "${METRICS}" > "${SUMMARY}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8'), delimiter='\t'))
vals = {r['arm']: r for r in rows if r['last_train_loss'] not in ('-', '')}
print('Proxy A/B (train-loss signal)')
if 'control' not in vals or 'a_xsa9' not in vals:
    print('No valid candidate data. PROMOTE: none')
    sys.exit(0)
ctrl = float(vals['control']['last_train_loss'])
cand = float(vals['a_xsa9']['last_train_loss'])
delta = cand - ctrl
print(f"control_train_loss={ctrl:.4f}")
print(f"a_xsa9_train_loss={cand:.4f}")
print(f"delta(a_xsa9-control)={delta:+.4f}")
if delta < -0.005:
    print('PROMOTE: a_xsa9 (proxy)')
else:
    print('PROMOTE: none (proxy)')
PY

cat "${METRICS}"
cat "${SUMMARY}"
echo "PROXY_ROOT=${ROOT}"
