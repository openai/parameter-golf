#!/usr/bin/env bash
set -euo pipefail

# Fast startup orchestration for 8xH100 RunPod runs.
# Strategy:
# 1) verify pod is live via RunPod API
# 2) sync tokenizer (blocking)
# 3) stage dataset in background
# 4) run synthetic precompile (max-autotune) into persistent caches
# 5) wait for dataset, then run smoke test using same cache

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

[[ -f "${ROOT_DIR}/.runpod_secret.sh" ]] && source "${ROOT_DIR}/.runpod_secret.sh"
RUNPOD_API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY or .runpod_secret.sh}"

POD_ID="${POD_ID:-}"
CACHE_ROOT="${CACHE_ROOT:-/workspace}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_8192_bpe.model}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp8192}"
TRAIN_FILES_GLOB="${DATA_PATH}/fineweb_train_*.bin"
VAL_FILES_GLOB="${DATA_PATH}/fineweb_val_*.bin"
HF_REPO="${HF_REPO:-kevclark/parameter-golf}"
DATA_SHARDS="${DATA_SHARDS:-1}"
ALLOW_NO_POD="${ALLOW_NO_POD:-0}"
MONITOR_POD="${MONITOR_POD:-1}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-30}"

# Optional commands you can still override at launch time.
TOKENIZER_FETCH_CMD="${TOKENIZER_FETCH_CMD:-}"
DATA_STAGE_CMD="${DATA_STAGE_CMD:-}"
SMOKE_CMD="${SMOKE_CMD:-python3 build_submission.py >/dev/null && python3 train_gpt.py}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

resolve_cache_root() {
  if mkdir -p "${CACHE_ROOT}" >/dev/null 2>&1; then
    return 0
  fi
  CACHE_ROOT="${ROOT_DIR}/.runtime_cache"
  mkdir -p "${CACHE_ROOT}"
  log "CACHE_ROOT fallback -> ${CACHE_ROOT}"
}

preflight_torch() {
  python3 - <<'PY'
import importlib.util, sys
ok = importlib.util.find_spec("torch") is not None
sys.exit(0 if ok else 1)
PY
}

POD_MONITOR_PID=""
DATA_STAGE_PID=""

check_runpod_pod() {
  local query response
  if [[ -n "${POD_ID}" ]]; then
    query=$(cat <<JSON
{"query":"query Pod { pod(input: {podId: \"${POD_ID}\"}) { id name desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }"}
JSON
)
  else
    query='{"query":"query Pods { myself { pods { id name desiredStatus runtime { uptimeInSeconds } } } }"}'
  fi

  response="$(curl -fsS 'https://api.runpod.io/graphql' \
    -H "Authorization: ${RUNPOD_API_KEY}" \
    -H 'Content-Type: application/json' \
    --data "${query}")"

  if [[ -n "${POD_ID}" ]]; then
    RESPONSE_JSON="${response}" python3 - <<'PY'
import json, os
j=json.loads(os.environ["RESPONSE_JSON"])
if j.get('errors'):
    raise SystemExit(f"RunPod API error: {j['errors']}")
pod=((j.get('data') or {}).get('pod') or {})
if not pod:
    raise SystemExit('Pod not found')
print(f"pod_ok:id={pod.get('id')} name={pod.get('name')} status={pod.get('desiredStatus')} uptime={((pod.get('runtime') or {}).get('uptimeInSeconds'))}")
PY
  else
    RESPONSE_JSON="${response}" python3 - <<'PY'
import json, os
j=json.loads(os.environ["RESPONSE_JSON"])
if j.get('errors'):
    raise SystemExit(f"RunPod API error: {j['errors']}")
pods=((j.get('data') or {}).get('myself') or {}).get('pods') or []
running=[p for p in pods if (p.get('desiredStatus') or '').upper()=='RUNNING']
print(f"pods_total:{len(pods)} running:{len(running)}")
for p in running[:10]:
    print(f"pod_ok:id={p.get('id')} name={p.get('name')} status={p.get('desiredStatus')} uptime={((p.get('runtime') or {}).get('uptimeInSeconds'))}")
PY
    local total running
    total="$(RESPONSE_JSON="${response}" python3 - <<'PY'
import json, os
j=json.loads(os.environ["RESPONSE_JSON"])
pods=((j.get("data") or {}).get("myself") or {}).get("pods") or []
print(len(pods))
PY
)"
    running="$(RESPONSE_JSON="${response}" python3 - <<'PY'
import json, os
j=json.loads(os.environ["RESPONSE_JSON"])
pods=((j.get("data") or {}).get("myself") or {}).get("pods") or []
print(sum(1 for p in pods if (p.get("desiredStatus") or "").upper() == "RUNNING"))
PY
)"
    if [[ "${running}" -eq 0 ]]; then
      if [[ "${ALLOW_NO_POD}" == "1" ]]; then
        log "No RUNNING pods found (pods_total=${total}); continuing because ALLOW_NO_POD=1."
      else
        log "No RUNNING pods found (pods_total=${total})."
        log "Start a pod first or set POD_ID; override with ALLOW_NO_POD=1 if you want to proceed anyway."
        return 1
      fi
    fi
  fi
}

pod_monitor_loop() {
  while true; do
    sleep "${MONITOR_INTERVAL_SEC}"
    local q
    q='query { myself { pods { id name desiredStatus runtime { uptimeInSeconds } } } }'
    local resp
    resp="$(curl -sS --request POST --header 'content-type: application/json' --header "Authorization: ${RUNPOD_API_KEY}" --url 'https://api.runpod.io/graphql' --data "$(jq -n --arg q "$q" '{"query": $q}')" || true)"
    RESPONSE_JSON="${resp}" python3 - <<'PY' || true
import json, os
s=os.environ.get("RESPONSE_JSON","").strip()
if not s:
    raise SystemExit(0)
j=json.loads(s)
pods=((j.get("data") or {}).get("myself") or {}).get("pods") or []
running=[p for p in pods if (p.get("desiredStatus") or "").upper()=="RUNNING"]
print(f"[pod-monitor] pods_total={len(pods)} running={len(running)}")
PY
  done
}

stage_tokenizer() {
  log "Ensuring tokenizer exists (blocking)..."
  mkdir -p "$(dirname "${TOKENIZER_PATH}")"
  if [[ -n "${TOKENIZER_FETCH_CMD}" ]]; then
    TOKENIZER_PATH="${TOKENIZER_PATH}" bash -lc "${TOKENIZER_FETCH_CMD}"
    return 0
  fi
  curl -L -s -o "$(dirname "${TOKENIZER_PATH}")/fineweb_8192_bpe.model" \
    "https://huggingface.co/datasets/kevclark/parameter-golf/resolve/main/datasets/tokenizers/fineweb_8192_bpe.model?download=true"
  curl -L -s -o "$(dirname "${TOKENIZER_PATH}")/fineweb_8192_bpe.vocab" \
    "https://huggingface.co/datasets/kevclark/parameter-golf/resolve/main/datasets/tokenizers/fineweb_8192_bpe.vocab?download=true"
  curl -L -s -o "$(dirname "${TOKENIZER_PATH}")/tokenizer_specs_8192.json" \
    "https://huggingface.co/datasets/kevclark/parameter-golf/resolve/main/datasets/tokenizers/tokenizer_specs_8192.json?download=true"
  [[ -f "$(dirname "${TOKENIZER_PATH}")/fineweb_8192_bpe.model" ]] || { log "Tokenizer download failed"; return 1; }
}

stage_dataset_async() {
  if compgen -G "${TRAIN_FILES_GLOB}" >/dev/null && compgen -G "${VAL_FILES_GLOB}" >/dev/null; then
    log "Dataset shards already present at ${DATA_PATH}; skipping data stage."
    return 0
  fi
  log "Starting dataset stage in background..."
  if [[ -n "${DATA_STAGE_CMD}" ]]; then
    bash -lc "${DATA_STAGE_CMD}" &
  else
    DATA_PATH="${DATA_PATH}" HF_REPO="${HF_REPO}" DATA_SHARDS="${DATA_SHARDS}" python3 - <<'PY' &
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
import os, pathlib, shutil, time

repo = os.environ["HF_REPO"]
data_dir = os.environ["DATA_PATH"]
num_shards = int(os.environ.get("DATA_SHARDS", "1"))
os.makedirs(data_dir, exist_ok=True)

def fetch_robust(filename, subfolders, dest_path):
    if os.path.exists(dest_path):
        return f"{filename}: cached"
    for subfolder in subfolders:
        for attempt in range(3):
            try:
                p = hf_hub_download(repo_id=repo, filename=filename, subfolder=subfolder, repo_type="dataset")
                src = pathlib.Path(p).resolve()
                try:
                    os.link(src, dest_path)
                except Exception:
                    shutil.copy2(src, dest_path)
                return f"{filename}: done"
            except Exception:
                if attempt == 2:
                    continue
                time.sleep(2 ** attempt)
    return f"{filename}: FAILED"

print(fetch_robust("fineweb_val_000000.bin", ["datasets/datasets/fineweb10B_sp8192", "datasets/fineweb10B_sp8192"], os.path.join(data_dir, "fineweb_val_000000.bin")), flush=True)

shards = [f"fineweb_train_{i:06d}.bin" for i in range(num_shards)]
with ThreadPoolExecutor(max_workers=min(4, max(num_shards, 1))) as ex:
    futs = [ex.submit(fetch_robust, s, ["datasets/datasets/fineweb10B_sp8192", "datasets/fineweb10B_sp8192"], os.path.join(data_dir, s)) for s in shards]
    for i, f in enumerate(as_completed(futs), start=1):
        print(f"[{i}/{len(shards)}] {f.result()}", flush=True)
PY
  fi
  DATA_STAGE_PID=$!
  export DATA_STAGE_PID
}

run_precompile() {
  log "Launching synthetic max-autotune precompile..."
  resolve_cache_root
  mkdir -p "${CACHE_ROOT}/torch_cache" "${CACHE_ROOT}/triton_cache"
  if ! preflight_torch; then
    log "PyTorch is not installed in this environment; cannot run precompile locally."
    return 1
  fi

  export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-64}"
  export TORCHINDUCTOR_FX_GRAPH_CACHE=1
  export TORCHINDUCTOR_AUTOGRAD_CACHE=1
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torch_cache}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton_cache}"

  export CURRICULUM_ENABLED="${CURRICULUM_ENABLED:-0}"
  export SEQ_LEN_START="${SEQ_LEN_START:-0}"
  export BATCH_TOKENS_START="${BATCH_TOKENS_START:-0}"

  COMPILE_MODE=max-autotune \
  COMPILER_WARMUP_STEPS="${COMPILER_WARMUP_STEPS:-1}" \
  PRECOMPILE_ONLY=1 \
  SYNTHETIC_WARMUP=1 \
  COMPILE_TRITON_CUDAGRAPHS="${COMPILE_TRITON_CUDAGRAPHS:-0}" \
  COMPILE_SHAPE_PADDING="${COMPILE_SHAPE_PADDING:-1}" \
  python3 build_submission.py >/dev/null && python3 train_gpt.py
}

wait_for_dataset() {
  if [[ -n "${DATA_STAGE_PID:-}" ]]; then
    log "Waiting for dataset stage PID ${DATA_STAGE_PID}..."
    wait "${DATA_STAGE_PID}"
  fi
}

run_smoke() {
  log "Running smoke test with cached compile artifacts..."
  resolve_cache_root
  if ! preflight_torch; then
    log "PyTorch is not installed in this environment; cannot run smoke test locally."
    return 1
  fi

  export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-64}"
  export TORCHINDUCTOR_FX_GRAPH_CACHE=1
  export TORCHINDUCTOR_AUTOGRAD_CACHE=1
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torch_cache}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton_cache}"

  ITERATIONS="${ITERATIONS:-2}" \
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}" \
  COMPILER_WARMUP_STEPS="${COMPILER_WARMUP_STEPS:-0}" \
  PRECOMPILE_ONLY=0 \
  SYNTHETIC_WARMUP=0 \
  bash -lc "${SMOKE_CMD}"
}

main() {
  log "Checking RunPod pod status..."
  check_runpod_pod
  if [[ "${MONITOR_POD}" == "1" ]]; then
    pod_monitor_loop &
    POD_MONITOR_PID=$!
  fi

  stage_tokenizer
  stage_dataset_async
  run_precompile
  wait_for_dataset
  run_smoke

  log "Done. Compile cache and smoke test succeeded."
}

cleanup() {
  local ec=$?
  if [[ -n "${DATA_STAGE_PID}" ]]; then
    kill "${DATA_STAGE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${POD_MONITOR_PID}" ]]; then
    kill "${POD_MONITOR_PID}" 2>/dev/null || true
  fi
  exit $ec
}
trap cleanup EXIT

main "$@"
