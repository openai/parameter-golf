#!/usr/bin/env bash
# vast_ngram_sweep.sh — Rent 8xH100 on Vast.ai, train podracer, sweep n-gram params
#
# Usage:
#   ./scripts/vast_ngram_sweep.sh
#   ./scripts/vast_ngram_sweep.sh --price 20.00
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
MAX_PRICE="${MAX_PRICE:-20.00}"
DISK_GB=80
SSH_KEY="$HOME/.ssh/id_ed25519_apollo"
IMAGE="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${LOCAL_DIR}/results/vast_ngram_sweeps"
POLL_INTERVAL=10
MAX_WAIT=600
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LABEL="ngram_sweep_${TIMESTAMP}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --price) MAX_PRICE="$2"; shift 2 ;;
        *)       echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  Vast.ai N-gram Sweep (8xH100)"
echo "  Label: $RUN_LABEL"
echo "  Max price: \$${MAX_PRICE}/hr"
echo "============================================"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
command -v vastai &>/dev/null || { echo "ERROR: vastai CLI not installed"; exit 1; }
[ -f "$SSH_KEY" ] || { echo "ERROR: SSH key not found at $SSH_KEY"; exit 1; }

# Verify local files exist
SOTA_DIR="${LOCAL_DIR}/concepts/podracer/sota_verified"
[ -f "${SOTA_DIR}/train_gpt.py" ] || { echo "ERROR: ${SOTA_DIR}/train_gpt.py not found"; exit 1; }
[ -f "${LOCAL_DIR}/concepts/podracer/sota/sweep_ngram.py" ] || { echo "ERROR: sweep_ngram.py not found"; exit 1; }
[ -f "${LOCAL_DIR}/concepts/podracer/podracer_red/run_safe.sh" ] || { echo "ERROR: run_safe.sh not found"; exit 1; }

# Check data exists locally
VAL_COUNT=$(ls "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_val_"*.bin 2>/dev/null | wc -l)
TRAIN_COUNT=$(ls "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_train_"*.bin 2>/dev/null | wc -l)
[ "$VAL_COUNT" -gt 0 ] || { echo "ERROR: No val shards found"; exit 1; }
[ "$TRAIN_COUNT" -gt 0 ] || { echo "ERROR: No train shards found"; exit 1; }
echo "==> Local data: ${TRAIN_COUNT} train shards, ${VAL_COUNT} val shards"
echo "==> SOTA file: ${SOTA_DIR}/train_gpt.py ($(md5sum "${SOTA_DIR}/train_gpt.py" | cut -d' ' -f1))"

# ── Find offer ────────────────────────────────────────────────────────────────
echo ""
echo "==> Searching for ${NUM_GPUS}x${GPU} (max \$${MAX_PRICE}/hr)..."

OFFER_JSON=$(vastai search offers \
    "gpu_name=${GPU} num_gpus=${NUM_GPUS} gpu_ram>=${MIN_VRAM} reliability>${MIN_RELIABILITY} rentable=True dph_total<=${MAX_PRICE} verified=True" \
    -t on-demand -o 'dph_total' --raw 2>/dev/null | head -1)

if [ -z "$OFFER_JSON" ] || [ "$OFFER_JSON" = "[]" ]; then
    echo "ERROR: No ${NUM_GPUS}x${GPU} offers under \$${MAX_PRICE}/hr"
    exit 1
fi

OFFER_ID=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'] if isinstance(d,list) else d['id'])")
OFFER_PRICE=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); e=d[0] if isinstance(d,list) else d; print(f\"{e['dph_total']:.2f}\")")

echo "==> Best offer: ID=${OFFER_ID}  ${NUM_GPUS}x${GPU}  \$${OFFER_PRICE}/hr"
echo ""
read -p "Rent this instance? ~80 min = ~\$$(python3 -c "print(f'{float('${OFFER_PRICE}') * 1.4:.0f}')") [y/N] " -n 1 -r
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
[ -z "$INSTANCE_ID" ] && { echo "ERROR: Could not parse instance ID"; exit 1; }
echo "==> Instance ID: $INSTANCE_ID"

# ── Wait for ready ────────────────────────────────────────────────────────────
echo "==> Waiting for instance..."
WAITED=0
STATUS="unknown"
while [ $WAITED -lt $MAX_WAIT ]; do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status','?'))" 2>/dev/null || echo "unknown")
    [ "$STATUS" = "running" ] && break
    echo "    status=$STATUS (${WAITED}s/${MAX_WAIT}s)"
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
[ -z "$SSH_PORT" ] || [ -z "$SSH_HOST" ] && { echo "ERROR: Bad SSH URL: $SSH_URL"; vastai destroy instance "$INSTANCE_ID"; exit 1; }

SSH_CMD="ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i $SSH_KEY -p $SSH_PORT $SSH_HOST"
SCP_CMD="scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i $SSH_KEY -P $SSH_PORT"

echo "==> Testing SSH ($SSH_HOST:$SSH_PORT)..."
RETRIES=0
while [ $RETRIES -lt 6 ]; do
    $SSH_CMD "echo OK" 2>/dev/null | grep -q OK && break
    RETRIES=$((RETRIES + 1)); sleep 5
done
[ $RETRIES -ge 6 ] && { echo "ERROR: SSH failed"; vastai destroy instance "$INSTANCE_ID"; exit 1; }
echo "    SSH OK"

# ── Build payload ─────────────────────────────────────────────────────────────
echo "==> Building payload..."
PAYLOAD_DIR=$(mktemp -d)
trap "rm -rf $PAYLOAD_DIR" EXIT

mkdir -p "$PAYLOAD_DIR/workspace/parameter-golf/concepts/podracer/sota"
mkdir -p "$PAYLOAD_DIR/workspace/parameter-golf/concepts/podracer/podracer_red"
mkdir -p "$PAYLOAD_DIR/workspace/parameter-golf/data/datasets/fineweb10B_sp1024"
mkdir -p "$PAYLOAD_DIR/workspace/parameter-golf/data/tokenizers"
mkdir -p "$PAYLOAD_DIR/workspace/parameter-golf/logs"

# SOTA verified train_gpt.py
cp "${SOTA_DIR}/train_gpt.py" "$PAYLOAD_DIR/workspace/parameter-golf/concepts/podracer/sota/"
cp "${SOTA_DIR}/train_gpt.py" "$PAYLOAD_DIR/workspace/parameter-golf/concepts/podracer/podracer_red/"

# Sweep script + run scripts
cp "${LOCAL_DIR}/concepts/podracer/sota/sweep_ngram.py" "$PAYLOAD_DIR/workspace/parameter-golf/concepts/podracer/sota/"
cp "${LOCAL_DIR}/concepts/podracer/podracer_red/run_safe.sh" "$PAYLOAD_DIR/workspace/parameter-golf/concepts/podracer/podracer_red/"
cp "${LOCAL_DIR}/concepts/podracer/sota/run_sweep.sh" "$PAYLOAD_DIR/workspace/parameter-golf/concepts/podracer/sota/"

# Data (train + val + tokenizer)
cp "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/"*.bin "$PAYLOAD_DIR/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/"
cp "${LOCAL_DIR}/data/tokenizers/fineweb_1024_bpe.model" "$PAYLOAD_DIR/workspace/parameter-golf/data/tokenizers/"

# Tarball
TARBALL="/tmp/vast_ngram_${TIMESTAMP}.tar.gz"
(cd "$PAYLOAD_DIR/workspace/parameter-golf" && tar czf "$TARBALL" .)
echo "==> Payload: $(du -sh "$TARBALL" | cut -f1)"

# ── Upload ────────────────────────────────────────────────────────────────────
echo "==> Uploading payload (this may take a few minutes)..."
$SCP_CMD "$TARBALL" "${SSH_HOST}:/workspace/payload.tar.gz"

echo "==> Extracting + installing deps..."
$SSH_CMD "
    mkdir -p /workspace/parameter-golf && cd /workspace/parameter-golf && tar xzf /workspace/payload.tar.gz &&
    pip install -q sentencepiece zstandard 2>&1 | tail -1 &&
    echo EXTRACT_OK
" 2>/dev/null

# Flash Attention 3
echo "==> Building Flash Attention 3..."
$SSH_CMD "
    cd /workspace/parameter-golf &&
    if python3 -c 'from flash_attn_interface import flash_attn_func' 2>/dev/null; then
        echo 'FA3 already available'
    else
        git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git 2>/dev/null || true
        cd flash-attention/hopper
        export FLASH_ATTENTION_DISABLE_FP16=TRUE
        export FLASH_ATTENTION_DISABLE_FP8=TRUE
        export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
        export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
        export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
        export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
        export FLASH_ATTENTION_DISABLE_SM80=TRUE
        export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
        export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
        export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
        export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
        export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
        export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
        export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
        export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
        export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
        export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
        pip install --no-build-isolation -e . 2>&1 | tail -5
        cd ../..
    fi
    python3 -c 'from flash_attn_interface import flash_attn_func; print(\"FA3 OK\")'
" 2>/dev/null

# ── Step 1: Train clean SOTA ─────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  STEP 1: Train clean SOTA (~10 min)"
echo "============================================"

$SSH_CMD "
    cd /workspace/parameter-golf &&
    export PYTHONPATH=/workspace/parameter-golf/flash-attention/hopper:\${PYTHONPATH:-} &&
    SEED=2045 \
    F1_CORR_RANK=0 \
    DISTILL_ENABLED=0 \
    MLP_ACT=leaky_relu_sq \
    MLP_LEAKY_SLOPE=0.5 \
    XSA_LAST_N=4 \
    BIGRAM_VOCAB_SIZE=1536 \
    ROPE_DIMS=24 \
    TTT_EVAL_ENABLED=0 \
    COMPILE_ENABLED=1 \
    COMPILE_FULLGRAPH=0 \
    NGRAM_EVAL_ORDER=7 \
    NGRAM_EVAL_MIN_ORDER=2 \
    NGRAM_EVAL_ADAPTIVE=1 \
    NGRAM_EVAL_ALPHA=0.30 \
    NGRAM_EVAL_ALPHA_MIN=0.05 \
    NGRAM_EVAL_ALPHA_MAX=0.60 \
    NGRAM_EVAL_ENTROPY_CENTER=4.0 \
    NGRAM_EVAL_ENTROPY_SCALE=2.0 \
    NGRAM_EVAL_MIN_COUNT=2 \
    NGRAM_EVAL_BUCKETS=4194304 \
    NGRAM_EVAL_MAX_SECONDS=300 \
    torchrun --standalone --nproc_per_node=8 \
        concepts/podracer/sota/train_gpt.py \
        2>&1
" 2>/dev/null | tee "/tmp/vast_train_${RUN_LABEL}.log"

echo ""
echo "==> Saving baseline model..."
$SSH_CMD "cd /workspace/parameter-golf && cp final_model.int6.ptz podracer_baseline.int6.ptz && ls -lh final_model.int6.ptz" 2>/dev/null

# ── Step 2: Sweep ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  STEP 2: N-gram sweep (~60 min)"
echo "============================================"

$SSH_CMD "
    cd /workspace/parameter-golf &&
    export PYTHONPATH=/workspace/parameter-golf/flash-attention/hopper:\${PYTHONPATH:-} &&
    SEED=2045 \
    MLP_ACT=leaky_relu_sq \
    MLP_LEAKY_SLOPE=0.5 \
    XSA_LAST_N=4 \
    BIGRAM_VOCAB_SIZE=1536 \
    ROPE_DIMS=24 \
    TTT_EVAL_ENABLED=0 \
    COMPILE_ENABLED=1 \
    COMPILE_FULLGRAPH=0 \
    MODEL_PATH=final_model.int6.ptz \
    SWEEP_MAX_SECONDS=180 \
    torchrun --standalone --nproc_per_node=8 \
        concepts/podracer/sota/sweep_ngram.py \
        2>&1
" 2>/dev/null | tee "/tmp/vast_sweep_${RUN_LABEL}.log"

# ── Pull results ──────────────────────────────────────────────────────────────
echo ""
echo "==> Pulling results..."
mkdir -p "$RESULTS_DIR"
$SCP_CMD "${SSH_HOST}:/workspace/parameter-golf/sweep_ngram_results.csv" "$RESULTS_DIR/sweep_${RUN_LABEL}.csv" 2>/dev/null || true
$SCP_CMD "${SSH_HOST}:/workspace/parameter-golf/podracer_baseline.int6.ptz" "$RESULTS_DIR/podracer_baseline_${RUN_LABEL}.int6.ptz" 2>/dev/null || true

cp "/tmp/vast_train_${RUN_LABEL}.log" "$RESULTS_DIR/" 2>/dev/null || true
cp "/tmp/vast_sweep_${RUN_LABEL}.log" "$RESULTS_DIR/" 2>/dev/null || true

# ── Destroy ───────────────────────────────────────────────────────────────────
echo ""
echo "==> Destroying instance $INSTANCE_ID..."
vastai destroy instance "$INSTANCE_ID"
echo "==> Destroyed. No further charges."

echo ""
echo "============================================"
echo "  DONE"
echo "  Results: $RESULTS_DIR/sweep_${RUN_LABEL}.csv"
echo "  Model:   $RESULTS_DIR/podracer_baseline_${RUN_LABEL}.int6.ptz"
echo "  Logs:    $RESULTS_DIR/"
echo "============================================"
