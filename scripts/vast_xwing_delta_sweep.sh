#!/usr/bin/env bash
# vast_xwing_delta_sweep.sh — Rent 8xH100 on Vast.ai, run X-WING cubric/ngram delta sweep.
#
# Modes:
#   1) Train + sweep (default):
#      ./scripts/vast_xwing_delta_sweep.sh --price 24.00 --grid delta12
#   2) Sweep only from a local .int6.ptz:
#      ./scripts/vast_xwing_delta_sweep.sh --skip-train --model checkpoints/f1_sota_20260324_final_model.int6.ptz
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_API_KEY
#   SSH key at ~/.ssh/id_ed25519_apollo registered on vast.ai

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
GPU="${GPU:-H100_SXM}"
NUM_GPUS=8
MIN_VRAM=80000
MIN_RELIABILITY=0.95
MAX_PRICE="${MAX_PRICE:-24.00}"
DISK_GB=100
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_apollo}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${LOCAL_DIR}/results/vast_xwing_delta"
POLL_INTERVAL=10
MAX_WAIT=600

SEED="${SEED:-1337}"
DELTA_GRID="${DELTA_GRID:-delta12}"                  # interaction4 | delta12
SWEEP_MAX_SECONDS="${SWEEP_MAX_SECONDS:-180}"
CUBRIC_CADENCE="${CUBRIC_CADENCE:-32}"
NGRAM_CHUNK_TOKENS="${NGRAM_CHUNK_TOKENS:-1048576}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"                        # 0=train+eval, 1=eval only
MODEL_PATH="${MODEL_PATH:-}"                         # local model path for --skip-train mode

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="xwing_delta_${TIMESTAMP}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --price) MAX_PRICE="$2"; shift 2 ;;
        --grid) DELTA_GRID="$2"; shift 2 ;;
        --sweep-seconds) SWEEP_MAX_SECONDS="$2"; shift 2 ;;
        --cadence) CUBRIC_CADENCE="$2"; shift 2 ;;
        --chunk-tokens) NGRAM_CHUNK_TOKENS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=1; shift 1 ;;
        --model) MODEL_PATH="$2"; SKIP_TRAIN=1; shift 2 ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  Vast.ai X-WING Delta Sweep (8xH100)"
echo "  Label: ${RUN_LABEL}"
echo "  Max price: \$${MAX_PRICE}/hr"
echo "  Mode: $([ "${SKIP_TRAIN}" = "1" ] && echo "SWEEP_ONLY" || echo "TRAIN_PLUS_SWEEP")"
echo "  Grid: ${DELTA_GRID}"
echo "  Sweep budget per n-gram arm: ${SWEEP_MAX_SECONDS}s"
echo "============================================"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
command -v vastai &>/dev/null || { echo "ERROR: vastai CLI not installed"; exit 1; }
[ -f "$HOME/.vast_api_key" ] || { echo "ERROR: Vast API key missing (~/.vast_api_key)"; exit 1; }
[ -f "$SSH_KEY" ] || { echo "ERROR: SSH key not found at $SSH_KEY"; exit 1; }

for file in \
    "${LOCAL_DIR}/concepts/xwing/train_gpt.py" \
    "${LOCAL_DIR}/concepts/xwing/run.sh" \
    "${LOCAL_DIR}/concepts/xwing/run_delta_sweep.sh" \
    "${LOCAL_DIR}/concepts/xwing/sweep_cubric_ngram_delta.py"
do
    [ -f "$file" ] || { echo "ERROR: Missing file: $file"; exit 1; }
done

VAL_COUNT=$(ls "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_val_"*.bin 2>/dev/null | wc -l)
TRAIN_COUNT=$(ls "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_train_"*.bin 2>/dev/null | wc -l)
[ "$VAL_COUNT" -gt 0 ] || { echo "ERROR: No val shards found"; exit 1; }
[ "${SKIP_TRAIN}" = "1" ] || [ "$TRAIN_COUNT" -gt 0 ] || { echo "ERROR: No train shards found"; exit 1; }
[ -f "${LOCAL_DIR}/data/tokenizers/fineweb_1024_bpe.model" ] || { echo "ERROR: tokenizer missing"; exit 1; }

LOCAL_MODEL_PATH=""
if [ "${SKIP_TRAIN}" = "1" ]; then
    if [ -z "${MODEL_PATH}" ]; then
        for candidate in \
            "${LOCAL_DIR}/final_model.int6.ptz" \
            "${LOCAL_DIR}/checkpoints/f1_sota_20260324_final_model.int6.ptz" \
            "${LOCAL_DIR}/checkpoints/podracing_20260325_final_model.int6.ptz"
        do
            if [ -f "$candidate" ]; then
                LOCAL_MODEL_PATH="$candidate"
                break
            fi
        done
    else
        for candidate in "${MODEL_PATH}" "${LOCAL_DIR}/${MODEL_PATH}" "${LOCAL_DIR}/checkpoints/${MODEL_PATH}"; do
            if [ -f "$candidate" ]; then
                LOCAL_MODEL_PATH="$candidate"
                break
            fi
        done
    fi
    [ -n "${LOCAL_MODEL_PATH}" ] || { echo "ERROR: --skip-train requested but model not found"; exit 1; }
    echo "==> Sweep-only model: ${LOCAL_MODEL_PATH} ($(ls -lh "${LOCAL_MODEL_PATH}" | awk '{print $5}'))"
fi

echo "==> Local data check: ${TRAIN_COUNT} train shards, ${VAL_COUNT} val shards"

# ── Find offer ────────────────────────────────────────────────────────────────
echo ""
echo "==> Searching for ${NUM_GPUS}x${GPU} offers (max \$${MAX_PRICE}/hr)..."
OFFER_JSON=$(vastai search offers \
    "gpu_name=${GPU} num_gpus=${NUM_GPUS} gpu_ram>=${MIN_VRAM} reliability>${MIN_RELIABILITY} rentable=True dph_total<=${MAX_PRICE} verified=True" \
    -t on-demand -o 'dph_total' --raw 2>/dev/null | head -1)
[ -n "$OFFER_JSON" ] && [ "$OFFER_JSON" != "[]" ] || { echo "ERROR: No matching offers"; exit 1; }

OFFER_ID=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'] if isinstance(d,list) else d['id'])")
OFFER_PRICE=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); e=d[0] if isinstance(d,list) else d; print(f\"{e['dph_total']:.2f}\")")
OFFER_GPU=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); e=d[0] if isinstance(d,list) else d; print(e.get('gpu_name','?'))")

echo "==> Best offer: ID=${OFFER_ID}  ${NUM_GPUS}x${OFFER_GPU}  \$${OFFER_PRICE}/hr"
echo ""
read -p "Rent this instance? [y/N] " -n 1 -r
echo ""
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

# ── Create instance ───────────────────────────────────────────────────────────
echo "==> Creating instance..."
CREATE_OUT=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh --direct \
    --label "$RUN_LABEL" 2>&1)
echo "$CREATE_OUT"

INSTANCE_ID=$(echo "$CREATE_OUT" | grep -oE 'new_contract["\s:]+[0-9]+' | grep -oE '[0-9]+' | head -1)
[ -z "$INSTANCE_ID" ] && INSTANCE_ID=$(echo "$CREATE_OUT" | grep -oE '[0-9]+' | head -1)
[ -n "$INSTANCE_ID" ] || { echo "ERROR: Could not parse instance ID"; exit 1; }
echo "==> Instance ID: $INSTANCE_ID"

# ── Wait for running ──────────────────────────────────────────────────────────
echo "==> Waiting for instance..."
WAITED=0
STATUS="unknown"
while [ $WAITED -lt $MAX_WAIT ]; do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status','?'))" 2>/dev/null || echo "unknown")
    [ "$STATUS" = "running" ] && break
    echo "    status=${STATUS} (${WAITED}s/${MAX_WAIT}s)"
    sleep $POLL_INTERVAL
    WAITED=$((WAITED + POLL_INTERVAL))
done
[ "$STATUS" = "running" ] || { echo "ERROR: Instance didn't start"; vastai destroy instance "$INSTANCE_ID"; exit 1; }
echo "==> Running!"

# ── SSH setup ─────────────────────────────────────────────────────────────────
sleep 5
SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
SSH_PORT=$(echo "$SSH_URL" | grep -oE '\-p [0-9]+' | awk '{print $2}')
SSH_HOST=$(echo "$SSH_URL" | grep -oE '[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+' | tail -1)
[ -n "$SSH_PORT" ] && [ -n "$SSH_HOST" ] || {
    echo "ERROR: Bad SSH URL: $SSH_URL"
    vastai destroy instance "$INSTANCE_ID"
    exit 1
}

SSH_CMD="ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i $SSH_KEY -p $SSH_PORT $SSH_HOST"
SCP_CMD="scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i $SSH_KEY -P $SSH_PORT"

echo "==> Testing SSH (${SSH_HOST}:${SSH_PORT})..."
RETRIES=0
while [ $RETRIES -lt 6 ]; do
    $SSH_CMD "echo OK" 2>/dev/null | grep -q OK && break
    RETRIES=$((RETRIES + 1))
    sleep 5
done
[ $RETRIES -lt 6 ] || { echo "ERROR: SSH failed"; vastai destroy instance "$INSTANCE_ID"; exit 1; }
echo "    SSH OK"

# ── Build payload ─────────────────────────────────────────────────────────────
echo "==> Building payload..."
PAYLOAD_DIR=$(mktemp -d)
trap "rm -rf $PAYLOAD_DIR" EXIT

mkdir -p "$PAYLOAD_DIR/workspace/parameter-golf/concepts/xwing"
mkdir -p "$PAYLOAD_DIR/workspace/parameter-golf/data/datasets/fineweb10B_sp1024"
mkdir -p "$PAYLOAD_DIR/workspace/parameter-golf/data/tokenizers"
mkdir -p "$PAYLOAD_DIR/workspace/parameter-golf/logs"

cp "${LOCAL_DIR}/concepts/xwing/train_gpt.py" "$PAYLOAD_DIR/workspace/parameter-golf/concepts/xwing/"
cp "${LOCAL_DIR}/concepts/xwing/run.sh" "$PAYLOAD_DIR/workspace/parameter-golf/concepts/xwing/"
cp "${LOCAL_DIR}/concepts/xwing/run_delta_sweep.sh" "$PAYLOAD_DIR/workspace/parameter-golf/concepts/xwing/"
cp "${LOCAL_DIR}/concepts/xwing/sweep_cubric_ngram_delta.py" "$PAYLOAD_DIR/workspace/parameter-golf/concepts/xwing/"
if [ "${SKIP_TRAIN}" = "0" ]; then
    cp "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/"*.bin "$PAYLOAD_DIR/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/"
else
    cp "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_val_"*.bin "$PAYLOAD_DIR/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/"
fi
cp "${LOCAL_DIR}/data/tokenizers/fineweb_1024_bpe.model" "$PAYLOAD_DIR/workspace/parameter-golf/data/tokenizers/"

REMOTE_MODEL_PATH="/workspace/parameter-golf/final_model.int6.ptz"
if [ "${SKIP_TRAIN}" = "1" ]; then
    cp "${LOCAL_MODEL_PATH}" "$PAYLOAD_DIR/workspace/parameter-golf/final_model.int6.ptz"
fi

TARBALL="/tmp/vast_xwing_delta_${TIMESTAMP}.tar.gz"
(cd "$PAYLOAD_DIR/workspace/parameter-golf" && tar czf "$TARBALL" .)
echo "==> Payload size: $(du -sh "$TARBALL" | cut -f1)"

# ── Upload and extract ────────────────────────────────────────────────────────
echo "==> Uploading payload (this may take a few minutes)..."
$SCP_CMD "$TARBALL" "${SSH_HOST}:/workspace/payload.tar.gz"

echo "==> Extracting + installing deps..."
$SSH_CMD "
    mkdir -p /workspace/parameter-golf &&
    cd /workspace/parameter-golf &&
    tar xzf /workspace/payload.tar.gz &&
    pip install -q sentencepiece zstandard 2>&1 | tail -1 &&
    echo EXTRACT_OK
" 2>/dev/null

# Flash Attention / compatibility shim
echo "==> Installing flash-attn interface..."
$SSH_CMD "python3 -c \"
import os, sys
shim = '''
try:
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    from torch.nn.functional import scaled_dot_product_attention as _sdpa
    def flash_attn_func(q, k, v, causal=False):
        q2 = q.transpose(1, 2)
        k2 = k.transpose(1, 2)
        v2 = v.transpose(1, 2)
        out = _sdpa(q2, k2, v2, is_causal=causal)
        return out.transpose(1, 2)
'''
site = [p for p in sys.path if 'site-packages' in p and os.path.isdir(p)][0]
with open(os.path.join(site, 'flash_attn_interface.py'), 'w') as f:
    f.write(shim)
print('flash_attn_interface shim installed')
\"" 2>/dev/null

# ── Step 1: optional training ────────────────────────────────────────────────
if [ "${SKIP_TRAIN}" = "0" ]; then
    echo ""
    echo "============================================"
    echo "  STEP 1: Train X-WING (~16-18 min)"
    echo "============================================"
    $SSH_CMD "
        cd /workspace/parameter-golf &&
        SEED=${SEED} \
        NPROC_PER_NODE=8 \
        CUBRIC_CADENCE=${CUBRIC_CADENCE} \
        NGRAM_CHUNK_TOKENS=${NGRAM_CHUNK_TOKENS} \
        bash concepts/xwing/run.sh 2>&1
    " | tee "/tmp/vast_train_${RUN_LABEL}.log"
else
    echo "==> SKIP_TRAIN=1, using uploaded model: ${REMOTE_MODEL_PATH}"
fi

# ── Step 2: delta sweep ──────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  STEP 2: Cubric × N-gram delta sweep"
echo "============================================"
$SSH_CMD "
    cd /workspace/parameter-golf &&
    MODEL_PATH=${REMOTE_MODEL_PATH} \
    DELTA_GRID=${DELTA_GRID} \
    SWEEP_MAX_SECONDS=${SWEEP_MAX_SECONDS} \
    CUBRIC_CADENCE=${CUBRIC_CADENCE} \
    NGRAM_CHUNK_TOKENS=${NGRAM_CHUNK_TOKENS} \
    NPROC_PER_NODE=8 \
    bash concepts/xwing/run_delta_sweep.sh 2>&1
" | tee "/tmp/vast_delta_${RUN_LABEL}.log"

# ── Pull outputs ──────────────────────────────────────────────────────────────
echo ""
echo "==> Pulling results..."
mkdir -p "$RESULTS_DIR"

$SCP_CMD "${SSH_HOST}:/workspace/parameter-golf/sweep_cubric_ngram_delta_results.csv" \
    "$RESULTS_DIR/sweep_${RUN_LABEL}.csv" 2>/dev/null || true
$SCP_CMD "${SSH_HOST}:/workspace/parameter-golf/sweep_cubric_ngram_delta_summary.json" \
    "$RESULTS_DIR/summary_${RUN_LABEL}.json" 2>/dev/null || true
if [ "${SKIP_TRAIN}" = "0" ]; then
    $SCP_CMD "${SSH_HOST}:/workspace/parameter-golf/final_model.int6.ptz" \
        "$RESULTS_DIR/final_model_${RUN_LABEL}.int6.ptz" 2>/dev/null || true
fi

cp "/tmp/vast_train_${RUN_LABEL}.log" "$RESULTS_DIR/" 2>/dev/null || true
cp "/tmp/vast_delta_${RUN_LABEL}.log" "$RESULTS_DIR/" 2>/dev/null || true

# ── Destroy instance ─────────────────────────────────────────────────────────
echo ""
echo "==> Destroying instance $INSTANCE_ID..."
vastai destroy instance "$INSTANCE_ID"
echo "==> Destroyed. No further charges."

echo ""
echo "============================================"
echo "  DONE"
echo "  CSV:   ${RESULTS_DIR}/sweep_${RUN_LABEL}.csv"
echo "  JSON:  ${RESULTS_DIR}/summary_${RUN_LABEL}.json"
if [ "${SKIP_TRAIN}" = "0" ]; then
echo "  Model: ${RESULTS_DIR}/final_model_${RUN_LABEL}.int6.ptz"
fi
echo "  Logs:  ${RESULTS_DIR}/"
echo "============================================"
