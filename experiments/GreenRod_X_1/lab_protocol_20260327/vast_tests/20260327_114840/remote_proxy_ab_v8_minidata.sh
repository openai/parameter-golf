#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="20260327_114840"
REPO="/workspace/parameter-golf-lab"
TEST_DIR="${REPO}/experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/${RUN_TAG}"
SOURCE_TRAIN="${REPO}/experiments/GreenRod_X_1/train_gpt.py"
SRC_DATA="${REPO}/data/datasets/fineweb10B_sp1024"
SRC_TOK="${REPO}/data/tokenizers/fineweb_1024_bpe.model"

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd "${REPO}"

python3 - <<'PY'
import torch
print("TORCH_READY", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${TEST_DIR}/proxy_ab_v8_${STAMP}"
MINI="${ROOT}/mini_data"
mkdir -p "${ROOT}/runs" "${MINI}" "${MINI}/tokenizers"

# Build tiny shards with valid headers (update token count + size coherently).
python3 - "${SRC_DATA}/fineweb_train_000000.bin" "${MINI}/fineweb_train_000000.bin" 262144 <<'PY'
import numpy as np
import sys
src, dst, n_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
header = np.fromfile(src, dtype="<i4", count=256)
if header.size != 256:
    raise SystemExit(f"bad header: {src}")
orig_tokens = int(header[2])
n = min(n_tokens, orig_tokens)
tokens = np.fromfile(src, dtype="<u2", count=n, offset=256 * np.dtype("<i4").itemsize)
header = header.copy()
header[2] = int(tokens.size)
with open(dst, "wb") as f:
    header.astype("<i4").tofile(f)
    tokens.astype("<u2", copy=False).tofile(f)
PY
python3 - "${SRC_DATA}/fineweb_val_000000.bin" "${MINI}/fineweb_val_000000.bin" 131072 <<'PY'
import numpy as np
import sys
src, dst, n_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
header = np.fromfile(src, dtype="<i4", count=256)
if header.size != 256:
    raise SystemExit(f"bad header: {src}")
orig_tokens = int(header[2])
n = min(n_tokens, orig_tokens)
tokens = np.fromfile(src, dtype="<u2", count=n, offset=256 * np.dtype("<i4").itemsize)
header = header.copy()
header[2] = int(tokens.size)
with open(dst, "wb") as f:
    header.astype("<i4").tofile(f)
    tokens.astype("<u2", copy=False).tofile(f)
PY
cp "${SRC_TOK}" "${MINI}/tokenizers/fineweb_1024_bpe.model"

METRICS="${ROOT}/metrics.tsv"
SUMMARY="${ROOT}/summary.txt"
echo -e "arm\tseed\tcap_step\tcap_val_bpb\trun_dir\tlog" > "${METRICS}"

extract_cap() {
  local log_path="$1"
  python3 - "$log_path" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace')
m = re.findall(r"step:(\d+)/\d+ val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
if not m:
    print("-\t-")
else:
    s,b = m[-1]
    print(f"{s}\t{b}")
PY
}

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
    export RUN_ID="proxyab8_${arm}_s${seed}_${STAMP}"
    export SEED="${seed}"
    export DATA_PATH="${MINI}"
    export TOKENIZER_PATH="${MINI}/tokenizers/fineweb_1024_bpe.model"
    export GDN_NUM_LAYERS=0
    export XSA_LAST_N="${xsa}"
    export NGRAM_EVAL_ORDER=0
    export TRIGRAM=0
    export NGRAM_ENTROPY_SHIFT=0
    export TRAIN_BATCH_TOKENS=8192
    export TRAIN_SEQ_LEN=512
    export EVAL_SEQ_LEN=512
    export ITERATIONS=30
    export WARMDOWN_ITERS=10
    export WARMUP_STEPS=2
    export MAX_WALLCLOCK_SECONDS=90
    export VAL_LOSS_EVERY=5
    export TRAIN_LOG_EVERY=5
    export SKIP_FINAL_EVAL=1
    export COMPILE_ENABLED=0
    export COMPILE_FULLGRAPH=0
    torchrun --standalone --nproc_per_node=1 "${rpy}" 2>&1 | tee "${rlog}"
  ) || true
  local step bpb
  read -r step bpb < <(extract_cap "${rlog}")
  echo -e "${arm}\t${seed}\t${step}\t${bpb}\t${rdir}\t${rlog}" >> "${METRICS}"
}

run_arm control 11
run_arm a_xsa9 9

python3 - "${METRICS}" > "${SUMMARY}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8'), delimiter='\t'))
vals = {r['arm']: r for r in rows if r['cap_val_bpb'] not in ('-', '')}
print('Mini-data A/B (val_bpb proxy)')
if 'control' not in vals or 'a_xsa9' not in vals:
    print('No valid candidate data. PROMOTE: none')
    sys.exit(0)
ctrl = float(vals['control']['cap_val_bpb'])
cand = float(vals['a_xsa9']['cap_val_bpb'])
delta = cand - ctrl
print(f"control_val_bpb={ctrl:.4f}")
print(f"a_xsa9_val_bpb={cand:.4f}")
print(f"delta(a_xsa9-control)={delta:+.4f}")
if delta <= -0.0100:
    print('PROMOTE: a_xsa9 (mini-data proxy)')
else:
    print('PROMOTE: none (mini-data proxy)')
PY

cat "${METRICS}"
cat "${SUMMARY}"
echo "PROXY_ROOT=${ROOT}"
