#!/usr/bin/env bash
set -euo pipefail

# Move 39 pod runner: late-frontier GATE_WINDOW / n-gram scout.
#
# Two different success bars matter:
#   default BASE_STACK=2018: PR #2018 GatedXSA/LQER-top1/in-timer ngram base,
#     scout seed 1337 first because it is the weak seed in that 42/1337/2026 set.
#   fallback BASE_STACK=2014: PR #2014 Progressive3k base, scout seed 0 first
#     because it is the weak seed in that 42/314/0 set.
#   fallback BASE_STACK=1953: PR #1953 longctx/no_qv base, scout seed 1234.
# Usage on an 8xH100 pod:
#   bash pod_move39_gate_scout.sh prepare
#   bash pod_move39_gate_scout.sh run_split 0 40 12 planA_2014_gate40_attn_smear12
#   bash pod_move39_gate_scout.sh run_split 42 40 12 planA_2014_gate40_attn_smear12
#   bash pod_move39_gate_scout.sh run_split 314 40 12 planA_2014_gate40_attn_smear12
#   BASE_STACK=1953 bash pod_move39_gate_scout.sh run_split 1234 40 12 fallback_1953_gate40
# Optional artifact headroom branch for added modules:
#   PATH_A_V3_SMALL=1 bash pod_move39_gate_scout.sh run 1234 32 planD_pathav3
# Cheap non-record smoke path:
#   bash pod_move39_gate_scout.sh prepare_smoke
#   bash pod_move39_gate_scout.sh run_smoke 1234 32 smoke_gate32

MODE="${1:-help}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/parameter-golf}"
OUT_ROOT="${OUT_ROOT:-$WORKSPACE/move39_outputs}"
BIGRAM_PATCH_SCRIPT="${BIGRAM_PATCH_SCRIPT:-$WORKSPACE/patch_bigramhash_1953.py}"
PATH_A_V3_PATCH_SCRIPT="${PATH_A_V3_PATCH_SCRIPT:-$WORKSPACE/patch_path_a_v3_small_1953.py}"
QAWARE_NGRAM_PATCH_SCRIPT="${QAWARE_NGRAM_PATCH_SCRIPT:-$WORKSPACE/patch_qaware_ngram_2018.py}"
BASE_STACK="${BASE_STACK:-2018}"
CASEOPS_ROOT="$REPO_DIR/data/datasets/fineweb10B_sp8192_caseops"
CASEOPS_DATA="$CASEOPS_ROOT/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
CASEOPS_TOKENIZER="$CASEOPS_ROOT/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"

export WORKSPACE REPO_DIR OUT_ROOT BASE_STACK

case "$BASE_STACK" in
  2018|pr2018)
    BASE_STACK="2018"
    BASE_PR=2018
    BASE_COMMIT="92d4fab3959108c7b8b05474810116b0e00fb7be"
    BASE_BRANCH="move39_pr2018_gate_qaware"
    RECORD_DIR_REL="records/track_10min_16mb/2026-04-30_GatedXSA_LQERTop1_IntimerNgramTTT"
    DEFAULT_TRAIN_SEQ_LEN=2048
    DEFAULT_ROPE_TRAIN_SEQ_LEN=2048
    DEFAULT_TRAIN_SEQ_SCHEDULE=""
    DEFAULT_EVAL_SEQ_LEN=2560
    DEFAULT_EVAL_STRIDE=64
    DEFAULT_TTT_EVAL_SEQ_LEN=2560
    DEFAULT_TTT_BATCH_SIZE=64
    DEFAULT_TTT_CHUNK_SIZE=48
    DEFAULT_TTT_SHORT_SCORE_FIRST_ENABLED=0
    DEFAULT_TTT_SHORT_DOC_LEN=2000
    DEFAULT_TTT_SHORT_CHUNK_SIZE=48
    DEFAULT_TTT_SHORT_SCORE_FIRST_STEPS=""
    DEFAULT_PHASED_TTT_NUM_PHASES=1
    DEFAULT_PHASED_TTT_PREFIX_DOCS=1000
    DEFAULT_TTT_LORA_RANK=80
    DEFAULT_LQER_TOP_K=1
    DEFAULT_NGRAM_TILT_ENABLED=1
    DEFAULT_NGRAM_HINT_PRECOMPUTE_OUTSIDE=0
    DEFAULT_GATED_XSA=1
    DEFAULT_SKYLIGHT_MUON=0
    DEFAULT_SEED_HINT="1337 then 42 then 2026"
    ;;
  2014|pr2014)
    BASE_STACK="2014"
    BASE_PR=2014
    BASE_COMMIT="c9843c97dc6d24b7a806ef3a51effa7b26d67a97"
    BASE_BRANCH="move39_pr2014_gate"
    RECORD_DIR_REL="records/track_10min_16mb/2026-04-30_SP8192_CaseOps_Progressive3k_ShortDocTTT"
    DEFAULT_TRAIN_SEQ_LEN=3072
    DEFAULT_ROPE_TRAIN_SEQ_LEN=3072
    DEFAULT_TRAIN_SEQ_SCHEDULE="1024@0.100,2048@0.700,3072@1.000"
    DEFAULT_EVAL_SEQ_LEN=3072
    DEFAULT_EVAL_STRIDE=1536
    DEFAULT_TTT_EVAL_SEQ_LEN=3072
    DEFAULT_TTT_BATCH_SIZE=24
    DEFAULT_TTT_CHUNK_SIZE=48
    DEFAULT_TTT_SHORT_SCORE_FIRST_ENABLED=1
    DEFAULT_TTT_SHORT_DOC_LEN=2000
    DEFAULT_TTT_SHORT_CHUNK_SIZE=24
    DEFAULT_TTT_SHORT_SCORE_FIRST_STEPS="256:8,2000:24"
    DEFAULT_PHASED_TTT_NUM_PHASES=1
    DEFAULT_PHASED_TTT_PREFIX_DOCS=2500
    DEFAULT_TTT_LORA_RANK=80
    DEFAULT_LQER_TOP_K=3
    DEFAULT_NGRAM_TILT_ENABLED=0
    DEFAULT_NGRAM_HINT_PRECOMPUTE_OUTSIDE=0
    DEFAULT_GATED_XSA=0
    DEFAULT_SKYLIGHT_MUON=0
    DEFAULT_SEED_HINT="0 then 42 then 314"
    ;;
  1953|pr1953)
    BASE_STACK="1953"
    BASE_PR=1953
    BASE_COMMIT="bd495c13ca42090260ec34c37575ded96a11dec2"
    BASE_BRANCH="move39_pr1953_gate"
    RECORD_DIR_REL="records/track_10min_16mb/2026-04-30_LongCtx_NoQV_QK525_on_1945_1.0586"
    DEFAULT_TRAIN_SEQ_LEN=2048
    DEFAULT_ROPE_TRAIN_SEQ_LEN=2048
    DEFAULT_TRAIN_SEQ_SCHEDULE=""
    DEFAULT_EVAL_SEQ_LEN=2560
    DEFAULT_EVAL_STRIDE=64
    DEFAULT_TTT_EVAL_SEQ_LEN=2560
    DEFAULT_TTT_BATCH_SIZE=64
    DEFAULT_TTT_CHUNK_SIZE=48
    DEFAULT_TTT_SHORT_SCORE_FIRST_ENABLED=0
    DEFAULT_TTT_SHORT_DOC_LEN=2000
    DEFAULT_TTT_SHORT_CHUNK_SIZE=48
    DEFAULT_TTT_SHORT_SCORE_FIRST_STEPS=""
    DEFAULT_PHASED_TTT_NUM_PHASES=3
    DEFAULT_PHASED_TTT_PREFIX_DOCS=2500
    DEFAULT_TTT_LORA_RANK=80
    DEFAULT_LQER_TOP_K=3
    DEFAULT_NGRAM_TILT_ENABLED=0
    DEFAULT_NGRAM_HINT_PRECOMPUTE_OUTSIDE=0
    DEFAULT_GATED_XSA=0
    DEFAULT_SKYLIGHT_MUON=0
    DEFAULT_SEED_HINT="1234 then 42 then 0"
    ;;
  *)
    printf '[move39][fatal] unknown BASE_STACK=%s; use 2018, 2014, or 1953\n' "$BASE_STACK" >&2
    exit 1
    ;;
esac

log() {
  printf '[move39] %s\n' "$*"
}

die() {
  printf '[move39][fatal] %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

disk_free_gb() {
  df -BG "$WORKSPACE" | awk 'NR==2 {gsub("G","",$4); print $4}'
}

install_min_deps() {
  need_cmd git
  need_cmd python3

  if ! command -v lrzip >/dev/null 2>&1; then
    log "Installing lrzip (needed by COMPRESSOR=pergroup)..."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends lrzip
  fi

  log "Installing small Python helpers if missing..."
  python3 -m pip install -q --upgrade huggingface_hub sentencepiece brotli python-minifier tqdm hf_transfer
}

checkout_repo() {
  mkdir -p "$WORKSPACE" "$OUT_ROOT"
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "Cloning parameter-golf into $REPO_DIR"
    git clone https://github.com/openai/parameter-golf.git "$REPO_DIR"
  fi

  cd "$REPO_DIR"
  git fetch origin "pull/${BASE_PR}/head:move39_pr${BASE_PR}" --quiet
  git checkout -B "$BASE_BRANCH" "$BASE_COMMIT"
  git reset --hard "$BASE_COMMIT" --quiet
  git clean -fd --quiet
  test -f "$RECORD_DIR_REL/train_gpt.py" || die "record train_gpt.py missing"
  git rev-parse HEAD | tee "$OUT_ROOT/base_commit.txt"
  printf '%s\n' "$BASE_STACK" > "$OUT_ROOT/base_stack.txt"
}

download_caseops_data() {
  log "Downloading/reusing public CaseOps dataset and tokenizer (~16.2 GB)..."
  export HF_HUB_ENABLE_HF_TRANSFER=1
  mkdir -p "$CASEOPS_ROOT"
  python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_dir = Path(os.environ["REPO_DIR"])
local_dir = repo_dir / "data" / "datasets" / "fineweb10B_sp8192_caseops"
patterns = [
    "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/*",
    "datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.*",
    "datasets/manifest.json",
]
snapshot_download(
    repo_id="romeerp/parameter-golf-caseops-v1",
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=str(local_dir),
    max_workers=16,
)
print(local_dir)
PY

  local ntrain
  ntrain="$(find "$CASEOPS_DATA" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l | tr -d ' ')"
  [[ "$ntrain" == "80" ]] || die "expected 80 train shards, found $ntrain in $CASEOPS_DATA"
  test -f "$CASEOPS_DATA/fineweb_val_000000.bin" || die "missing val shard"
  test -f "$CASEOPS_DATA/fineweb_val_bytes_000000.bin" || die "missing val byte sidecar"
  test -f "$CASEOPS_TOKENIZER" || die "missing tokenizer $CASEOPS_TOKENIZER"
}

download_caseops_smoke_data() {
  log "Downloading/reusing minimal CaseOps smoke data (one train shard + val + tokenizer)..."
  export HF_HUB_ENABLE_HF_TRANSFER=1
  mkdir -p "$CASEOPS_ROOT"
  python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_dir = Path(os.environ["REPO_DIR"])
local_dir = repo_dir / "data" / "datasets" / "fineweb10B_sp8192_caseops"
patterns = [
    "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_000000.bin",
    "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_000000.bin",
    "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_000000.bin",
    "datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.*",
    "datasets/manifest.json",
]
snapshot_download(
    repo_id="romeerp/parameter-golf-caseops-v1",
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=str(local_dir),
    max_workers=8,
)
print(local_dir)
PY

  test -f "$CASEOPS_DATA/fineweb_train_000000.bin" || die "missing smoke train shard"
  test -f "$CASEOPS_DATA/fineweb_val_000000.bin" || die "missing val shard"
  test -f "$CASEOPS_DATA/fineweb_val_bytes_000000.bin" || die "missing val byte sidecar"
  test -f "$CASEOPS_TOKENIZER" || die "missing tokenizer $CASEOPS_TOKENIZER"
}

preflight() {
  log "Preflight starting..."
  install_min_deps
  checkout_repo

  local free_gb
  free_gb="$(disk_free_gb)"
  log "Free disk at $WORKSPACE: ${free_gb} GB"
  (( free_gb >= 35 )) || die "need at least 35 GB free before data download"

  download_caseops_data

  local gpu_count
  gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
  log "GPU count: $gpu_count"
  [[ "$gpu_count" == "8" ]] || die "need exactly 8 GPUs for record run"

  python3 - <<'PY'
import importlib
mods = ["torch", "sentencepiece", "brotli", "triton", "flash_attn_interface"]
for m in mods:
    importlib.import_module(m)
print("python_imports_ok")
PY

  cd "$REPO_DIR/$RECORD_DIR_REL"
  python3 -m py_compile train_gpt.py
  log "Preflight complete."
}

preflight_smoke() {
  log "Smoke preflight starting (non-record, cheap-pod only)..."
  install_min_deps
  checkout_repo

  local free_gb
  free_gb="$(disk_free_gb)"
  log "Free disk at $WORKSPACE: ${free_gb} GB"
  (( free_gb >= 5 )) || die "need at least 5 GB free before smoke data download"

  download_caseops_smoke_data

  local gpu_count
  gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
  log "GPU count: $gpu_count"
  (( gpu_count >= 1 )) || die "need at least 1 GPU for smoke"

  python3 - <<'PY'
import importlib
mods = ["torch", "sentencepiece", "brotli", "triton", "flash_attn_interface"]
for m in mods:
    importlib.import_module(m)
print("python_imports_ok")
PY

  cd "$REPO_DIR/$RECORD_DIR_REL"
  python3 -m py_compile train_gpt.py
  log "Smoke preflight complete."
}

parse_log() {
  local log_path="$1"
  python3 - "$log_path" <<'PY'
import json, re, sys
from pathlib import Path
p = Path(sys.argv[1])
txt = p.read_text(errors="replace")
def last(pattern, cast=float):
    ms = re.findall(pattern, txt)
    return cast(ms[-1]) if ms else None
out = {
    "log": str(p),
    "train_ms": last(r"stopping_early: wallclock_cap train_time: ([0-9]+)ms", int),
    "prequant_bpb": last(r"diagnostic pre-quantization post-ema val_loss:[0-9.]+ val_bpb:([0-9.]+)"),
    "quant_bpb": last(r"diagnostic quantized val_loss:[0-9.]+ val_bpb:([0-9.]+)"),
    "final_bpb": last(r"quantized_ttt_phased val_loss:[0-9.]+ val_bpb:([0-9.]+)"),
    "ttt_eval_ms": last(r"quantized_ttt_phased val_loss:[0-9.]+ val_bpb:[0-9.]+ eval_time:([0-9]+)ms", int),
    "artifact_bytes": last(r"Total submission size quantized\+pergroup: ([0-9]+)(?: bytes)?", int),
}
print(json.dumps(out, indent=2, sort_keys=True))
final = out["final_bpb"]
artifact = out["artifact_bytes"]
train = out["train_ms"]
eval_ms = out["ttt_eval_ms"]
stack = __import__("os").environ.get("BASE_STACK", "2014")
if artifact is not None and artifact >= 16000000:
    print("DECISION=HARD_STOP_ARTIFACT")
elif train is not None and train >= 600000:
    print("DECISION=HARD_STOP_TRAIN")
elif eval_ms is not None and eval_ms >= 600000:
    print("DECISION=HARD_STOP_EVAL")
elif final is None:
    print("DECISION=NO_FINAL_BPB")
elif stack == "2018":
    # PR #2018 weak seed 1337 baseline is 1.04721994. A real record over
    # #2018 needs a broad mean gain, so seed 1337 must move visibly.
    if final <= 1.04493:
        print("DECISION=MERGE_MARGIN_GO_OVER_2018_SEED1337")
    elif final <= 1.04572:
        print("DECISION=STRONG_GO_OVER_2018_SEED1337")
    elif final <= 1.04642:
        print("DECISION=VISIBLE_GO_OVER_2018_SEED1337")
    elif final <= 1.04695:
        print("DECISION=CONDITIONAL_NEAR_MISS_2018_SEED1337")
    else:
        print("DECISION=NO_GO")
elif stack == "2014":
    # PR #2014 seed-0 baseline is 1.05807084. These are scout thresholds for
    # the weak seed; final decisions still require the full 42/314/0 mean.
    if final <= 1.05578604:
        print("DECISION=MERGE_MARGIN_GO_OVER_2014_SEED0")
    elif final <= 1.05757084:
        print("DECISION=STRONG_GO_OVER_2014_SEED0")
    elif final <= 1.05777084:
        print("DECISION=VISIBLE_GO_OVER_2014_SEED0")
    elif final <= 1.05795084:
        print("DECISION=CONDITIONAL_NEAR_MISS_2014_SEED0")
    else:
        print("DECISION=NO_GO")
elif stack == "1953":
    if final <= 1.05666796:
        print("DECISION=MERGE_GO_OVER_1953")
    elif final <= 1.05870:
        print("DECISION=VISIBLE_STRONG_GO_NOT_1953_THRESHOLD")
    elif final <= 1.05883604:
        print("DECISION=VISIBLE_GO_NOT_1953_THRESHOLD")
    elif final <= 1.05890:
        print("DECISION=CONDITIONAL_NEAR_MISS")
    else:
        print("DECISION=NO_GO")
else:
    print("DECISION=NO_GO")
PY
}

run_seed() {
  local seed="${1:?seed required}"
  local gate="${2:?gate window required}"
  local label="${3:-gate${gate}}"
  local smear_gate="${4:-}"
  checkout_repo

  [[ -d "$CASEOPS_DATA" ]] || die "caseops data missing; run prepare first"
  [[ -f "$CASEOPS_TOKENIZER" ]] || die "caseops tokenizer missing; run prepare first"

  local run_id="move39_${label}_seed${seed}_$(date +%Y%m%d_%H%M%S)"
  local out_dir="$OUT_ROOT/$run_id"
  mkdir -p "$out_dir"

  cd "$REPO_DIR/$RECORD_DIR_REL"
  if [[ -n "$smear_gate" ]]; then
    log "Applying split-gate patch: sparse/attention GATE_WINDOW=$gate, SMEAR_GATE_WINDOW=$smear_gate"
    python3 - <<'PY'
import os
from pathlib import Path

p = Path("train_gpt.py")
text = p.read_text()
needle = '    gate_window = int(os.environ.get("GATE_WINDOW", 12))\n'
insert = (
    '    gate_window = int(os.environ.get("GATE_WINDOW", 12))\n'
    '    smear_gate_window = int(os.environ.get("SMEAR_GATE_WINDOW", gate_window))\n'
)
if "smear_gate_window = int(os.environ.get" not in text:
    if needle not in text:
        raise SystemExit("could not find gate_window definition")
    text = text.replace(needle, insert, 1)
old = "            self.smear_window = h.gate_window\n"
new = "            self.smear_window = h.smear_gate_window\n"
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise SystemExit("could not find SmearGate h.gate_window assignment")
p.write_text(text)
PY
    python3 -m py_compile train_gpt.py
  fi
  if [[ "${BIGRAMHASH_PATCH:-0}" == "1" || "${BIGRAM_VOCAB_SIZE:-0}" != "0" ]]; then
    log "Applying BigramHash patch: BIGRAM_VOCAB_SIZE=${BIGRAM_VOCAB_SIZE:-2048} BIGRAM_DIM=${BIGRAM_DIM:-16}"
    if [[ ! -f "$BIGRAM_PATCH_SCRIPT" ]]; then
      die "missing BigramHash patch script at $BIGRAM_PATCH_SCRIPT; copy playbook/patch_bigramhash_1953.py to the pod first"
    fi
    python3 "$BIGRAM_PATCH_SCRIPT" train_gpt.py
    python3 -m py_compile train_gpt.py
  fi
  if [[ "${PATH_A_V3_SMALL:-0}" == "1" ]]; then
    log "Applying Path A v3 small/control-tensor int8 patch"
    if [[ ! -f "$PATH_A_V3_PATCH_SCRIPT" ]]; then
      die "missing Path A v3 patch script at $PATH_A_V3_PATCH_SCRIPT; copy playbook/patch_path_a_v3_small_1953.py to the pod first"
    fi
    python3 "$PATH_A_V3_PATCH_SCRIPT" train_gpt.py
    python3 -m py_compile train_gpt.py
  fi
  if [[ "${QAWARE_NGRAM_PATCH:-0}" == "1" || "${NGRAM_QAWARE_DYNAMIC:-0}" == "1" ]]; then
    log "Applying q-aware dynamic n-gram tilt patch"
    if [[ ! -f "$QAWARE_NGRAM_PATCH_SCRIPT" ]]; then
      die "missing q-aware n-gram patch script at $QAWARE_NGRAM_PATCH_SCRIPT; copy playbook/patch_qaware_ngram_2018.py to the pod first"
    fi
    python3 "$QAWARE_NGRAM_PATCH_SCRIPT" .
    python3 -m py_compile train_gpt.py online_ngram_tilt.py
  fi
  if [[ "${SMOKE_MODE:-0}" == "1" ]]; then
    log "Applying NON-RECORD smoke patch: truncate validation to SMOKE_VAL_TOKENS=${SMOKE_VAL_TOKENS:-8192}"
    python3 - <<'PY'
import os
from pathlib import Path

p = Path("train_gpt.py")
text = p.read_text()
needle = "    val_data = ValidationData(h, device)\n"
insert = """    val_data = ValidationData(h, device)\n    smoke_val_tokens = int(os.environ.get("SMOKE_VAL_TOKENS", "0"))\n    if smoke_val_tokens > 0:\n        n = min(smoke_val_tokens + 1, val_data.val_tokens.numel())\n        val_data.val_tokens = val_data.val_tokens[:n].contiguous()\n        if getattr(val_data, "val_bytes", None) is not None:\n            val_data.val_bytes = val_data.val_bytes[:n].contiguous()\n        log(f"SMOKE_VAL_TOKENS active: truncated validation to {val_data.val_tokens.numel()-1} scored tokens")\n"""
if "SMOKE_VAL_TOKENS active" not in text:
    if needle not in text:
        raise SystemExit("could not find ValidationData initialization")
    text = text.replace(needle, insert, 1)
p.write_text(text)
PY
    python3 -m py_compile train_gpt.py
  fi
  if [[ "${PACK_CODE:-0}" == "1" ]]; then
    log "Applying optional source-code packing (artifact headroom branch; not default Plan A)"
    cp train_gpt.py "$out_dir/train_gpt_unpacked_before_pack.py"
    python3 - <<'PY'
import base64
from pathlib import Path

p = Path("train_gpt.py")
raw = p.read_bytes()
try:
    import brotli
    comp = brotli.compress(raw, quality=11, lgwin=24)
    b85 = base64.b85encode(comp).decode("ascii")
    stub = (
        "import brotli,base64\n"
        f"exec(brotli.decompress(base64.b85decode({b85!r})))\n"
    )
except Exception:
    import lzma
    comp = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    b85 = base64.b85encode(comp).decode("ascii")
    stub = (
        "import lzma,base64\n"
        f"exec(lzma.decompress(base64.b85decode({b85!r})))\n"
    )
p.write_text(stub, encoding="utf-8")
print(f"packed_code raw={len(raw)} compressed={len(comp)} stub={len(stub.encode())} saved={len(raw)-len(stub.encode())}")
PY
    python3 -m py_compile train_gpt.py
  fi
  log "Starting seed=$seed GATE_WINDOW=$gate label=$label"
  log "Base stack: PR #$BASE_PR ($BASE_STACK) commit=$BASE_COMMIT record_dir=$RECORD_DIR_REL"
  log "Output: $out_dir"

  env | sort > "$out_dir/env_before.txt"
  git rev-parse HEAD > "$out_dir/git_head.txt"
  git diff > "$out_dir/git_diff.patch"
  cp train_gpt.py "$out_dir/train_gpt.py"
  for helper in online_ngram_tilt.py online_ngram_state.c run.sh submission.json; do
    [[ -f "$helper" ]] && cp "$helper" "$out_dir/$helper"
  done

  set +e
  local nproc="${NPROC:-8}"
  NCCL_NET=Socket \
  OMP_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  RUN_ID="$run_id" \
  ARTIFACT_DIR="$out_dir" \
  SEED="$seed" \
  CASEOPS_ENABLED=1 \
  DATA_PATH="$CASEOPS_DATA" \
  TOKENIZER_PATH="$CASEOPS_TOKENIZER" \
  VOCAB_SIZE=8192 \
  ITERATIONS="${ITERATIONS:-20000}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-$DEFAULT_TRAIN_SEQ_LEN}" \
  ROPE_TRAIN_SEQ_LEN="${ROPE_TRAIN_SEQ_LEN:-$DEFAULT_ROPE_TRAIN_SEQ_LEN}" \
  TRAIN_SEQ_SCHEDULE="${TRAIN_SEQ_SCHEDULE:-$DEFAULT_TRAIN_SEQ_SCHEDULE}" \
  TRAIN_SEQ_SCHEDULE_MODE="${TRAIN_SEQ_SCHEDULE_MODE:-wallclock}" \
  SEQ_CHANGE_WARMUP_STEPS="${SEQ_CHANGE_WARMUP_STEPS:-32}" \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}" \
  VAL_BATCH_TOKENS="${VAL_BATCH_TOKENS:-524288}" \
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
  WARMDOWN_FRAC=0.85 \
  BETA2=0.99 \
  EMA_DECAY="${EMA_DECAY:-0.9965}" \
  TTT_ENABLED=1 \
  PHASED_TTT_ENABLED=1 \
  PHASED_TTT_NUM_PHASES="${PHASED_TTT_NUM_PHASES:-$DEFAULT_PHASED_TTT_NUM_PHASES}" \
  PHASED_TTT_PREFIX_DOCS="${PHASED_TTT_PREFIX_DOCS:-$DEFAULT_PHASED_TTT_PREFIX_DOCS}" \
  TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-$DEFAULT_TTT_BATCH_SIZE}" \
  TTT_CHUNK_SIZE="${TTT_CHUNK_SIZE:-$DEFAULT_TTT_CHUNK_SIZE}" \
  TTT_SHORT_SCORE_FIRST_ENABLED="${TTT_SHORT_SCORE_FIRST_ENABLED:-$DEFAULT_TTT_SHORT_SCORE_FIRST_ENABLED}" \
  TTT_SHORT_DOC_LEN="${TTT_SHORT_DOC_LEN:-$DEFAULT_TTT_SHORT_DOC_LEN}" \
  TTT_SHORT_CHUNK_SIZE="${TTT_SHORT_CHUNK_SIZE:-$DEFAULT_TTT_SHORT_CHUNK_SIZE}" \
  TTT_SHORT_SCORE_FIRST_STEPS="${TTT_SHORT_SCORE_FIRST_STEPS:-$DEFAULT_TTT_SHORT_SCORE_FIRST_STEPS}" \
  TTT_LORA_RANK="${TTT_LORA_RANK:-$DEFAULT_TTT_LORA_RANK}" \
  TTT_LORA_LR="${TTT_LORA_LR:-0.0001}" \
  TTT_BETA2=0.99 \
  TTT_WEIGHT_DECAY=0.5 \
  TTT_MASK=no_qv \
  TTT_Q_LORA=0 \
  TTT_V_LORA=0 \
  TTT_LOCAL_LR_MULT=0.75 \
  EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-$DEFAULT_EVAL_SEQ_LEN}" \
  EVAL_STRIDE="${EVAL_STRIDE:-$DEFAULT_EVAL_STRIDE}" \
  EVAL_INCLUDE_TAIL="${EVAL_INCLUDE_TAIL:-1}" \
  TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-$DEFAULT_TTT_EVAL_SEQ_LEN}" \
  QK_GAIN_INIT=5.25 \
  MATRIX_LR=0.026 \
  MIN_LR=0.1 \
  EMBED_BITS=7 \
  BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-0}" \
  BIGRAM_DIM="${BIGRAM_DIM:-32}" \
  BIGRAM_BITS="${BIGRAM_BITS:-6}" \
  MATRIX_CLIP_SIGMAS=12.85 \
  ATTN_CLIP_SIGMAS=13.0 \
  MLP_CLIP_SIGMAS=11.5 \
  EMBED_CLIP_SIGMAS=14.0 \
  GRAD_CLIP_NORM=0.3 \
  FUSED_CE_ENABLED=1 \
  SMEAR_GATE_ENABLED=1 \
  GATE_WINDOW="$gate" \
  SMEAR_GATE_WINDOW="${smear_gate:-$gate}" \
  SPARSE_ATTN_GATE_ENABLED=1 \
  SPARSE_ATTN_GATE_SCALE=0.5 \
  GATED_ATTN_QUANT_GATE=1 \
  PATH_A_V3_SMALL="${PATH_A_V3_SMALL:-0}" \
  LQER_ENABLED=1 \
  LQER_RANK=4 \
  LQER_TOP_K="${LQER_TOP_K:-$DEFAULT_LQER_TOP_K}" \
  LQER_GROUP_SIZE=64 \
  LQER_ASYM_ENABLED=1 \
  LQER_ASYM_GROUP=64 \
  AWQ_LITE_ENABLED=1 \
  ASYM_LOGIT_RESCALE=1 \
  NGRAM_TILT_ENABLED="${NGRAM_TILT_ENABLED:-$DEFAULT_NGRAM_TILT_ENABLED}" \
  NGRAM_HINT_PRECOMPUTE_OUTSIDE="${NGRAM_HINT_PRECOMPUTE_OUTSIDE:-$DEFAULT_NGRAM_HINT_PRECOMPUTE_OUTSIDE}" \
  NGRAM_QAWARE_DYNAMIC="${NGRAM_QAWARE_DYNAMIC:-0}" \
  NGRAM_QAWARE_GAIN_FLOOR="${NGRAM_QAWARE_GAIN_FLOOR:-0.0}" \
  GATED_XSA="${GATED_XSA:-$DEFAULT_GATED_XSA}" \
  SKYLIGHT_MUON="${SKYLIGHT_MUON:-$DEFAULT_SKYLIGHT_MUON}" \
  GPTQ_RESERVE_SECONDS="${GPTQ_RESERVE_SECONDS:-4.0}" \
  GPTQ_CALIBRATION_BATCHES="${GPTQ_CALIBRATION_BATCHES:-16}" \
  COMPRESSOR=pergroup \
  torchrun --standalone --nproc_per_node="$nproc" train_gpt.py 2>&1 | tee "$out_dir/torchrun_console.log"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "$status" > "$out_dir/exit_status.txt"

  local main_log="$out_dir/$run_id.txt"
  if [[ -f "$main_log" ]]; then
    parse_log "$main_log" | tee "$out_dir/metrics_and_decision.jsonish"
  else
    log "Main log not found at $main_log; parsing console log instead."
    parse_log "$out_dir/torchrun_console.log" | tee "$out_dir/metrics_and_decision.jsonish"
  fi

  log "Run finished with exit status $status"
  return "$status"
}

run_smoke() {
  local seed="${1:?seed required}"
  local gate="${2:?gate window required}"
  local label="${3:-smoke_gate${gate}}"
  SMOKE_MODE=1 \
  NPROC="${NPROC:-1}" \
  ITERATIONS="${ITERATIONS:-200}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-90}" \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}" \
  VAL_BATCH_TOKENS="${VAL_BATCH_TOKENS:-65536}" \
  EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-512}" \
  TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-512}" \
  PHASED_TTT_PREFIX_DOCS="${PHASED_TTT_PREFIX_DOCS:-10}" \
  GPTQ_RESERVE_SECONDS="${GPTQ_RESERVE_SECONDS:-4.0}" \
  GPTQ_CALIBRATION_BATCHES="${GPTQ_CALIBRATION_BATCHES:-1}" \
  SMOKE_VAL_TOKENS="${SMOKE_VAL_TOKENS:-8192}" \
  run_seed "$seed" "$gate" "$label"
}

case "$MODE" in
  prepare)
    preflight
    ;;
  prepare_smoke)
    preflight_smoke
    ;;
  run)
    run_seed "${2:-}" "${3:-}" "${4:-}"
    ;;
  run_smoke)
    run_smoke "${2:-}" "${3:-}" "${4:-}"
    ;;
  run_split)
    run_seed "${2:-}" "${3:-}" "${5:-split_g${3:-}_s${4:-}}" "${4:-}"
    ;;
  parse)
    parse_log "${2:?log path required}"
    ;;
  help|*)
    cat <<EOF
Usage:
  BASE_STACK=2018 bash $0 prepare       # default; PR #2018 strict in-timer ngram base
  BASE_STACK=2014 bash $0 prepare       # default; PR #2014 Progressive3k base
  BASE_STACK=1953 bash $0 prepare       # fallback; PR #1953 longctx/no_qv base
  bash $0 prepare
  bash $0 prepare_smoke
  bash $0 run <seed> <gate_window> <label>
  bash $0 run_smoke <seed> <gate_window> <label>
  bash $0 run_split <seed> <sparse_gate_window> <smear_gate_window> <label>
  bash $0 parse <log_path>

Recommended record sequence:
  bash $0 prepare
  QAWARE_NGRAM_PATCH=1 NGRAM_QAWARE_DYNAMIC=1 bash $0 run_split 1337 32 12 planA_2018_qaware_gate32
  # Continue only if seed 1337 is at least VISIBLE_GO and artifact/train/eval all clean:
  QAWARE_NGRAM_PATCH=1 NGRAM_QAWARE_DYNAMIC=1 bash $0 run_split 42 32 12 planA_2018_qaware_gate32
  QAWARE_NGRAM_PATCH=1 NGRAM_QAWARE_DYNAMIC=1 bash $0 run_split 2026 32 12 planA_2018_qaware_gate32

Fallback if #2018 is invalidated:
  BASE_STACK=2014 bash $0 prepare
  BASE_STACK=2014 bash $0 run_split 0 40 12 planB_2014_gate40_attn_smear12

Fallback to older #1953 base only if #2014 proves flawed:
  BASE_STACK=1953 bash $0 prepare
  BASE_STACK=1953 bash $0 run_split 1234 40 12 fallback_1953_gate40_attn_smear12

Fallbacks:
  # Only if Plan A is close but not enough, and artifact/train/eval margins are clean:
  BIGRAMHASH_PATCH=1 BIGRAM_VOCAB_SIZE=512 BIGRAM_DIM=8 BIGRAM_BITS=6 bash $0 run_split 0 40 12 planB_2014_gate40_bigram512d8

  # Safer but weaker Bigram fallback if 512x8 breaches artifact or timing:
  BIGRAMHASH_PATCH=1 BIGRAM_VOCAB_SIZE=512 BIGRAM_DIM=4 BIGRAM_BITS=6 bash $0 run_split 0 40 12 planC_2014_gate40_bigram512d4

  # Not recommended by A40 smoke, kept only as emergency artifact experiment:
  PATH_A_V3_SMALL=1 bash $0 run_split 0 40 12 planZ_pathav3_small

Cheap-pod smoke (NOT record evidence):
  bash $0 prepare_smoke
  bash $0 run_smoke 1234 32 smoke_gate32

Current default BASE_STACK=$BASE_STACK; recommended seed order: $DEFAULT_SEED_HINT.
EOF
    ;;
esac
