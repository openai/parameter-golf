#!/usr/bin/env bash
# vast_sweep.sh — Rent a Vast.ai GPU, run TTT sweep, pull results, destroy instance
#
# Usage:
#   ./scripts/vast_sweep.sh [--grid untested5] [--gpu H100_SXM] [--ptz final_model.int6.ptz]
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_API_KEY
#   SSH key registered at https://cloud.vast.ai/account/

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID="${GRID:-untested5}"
GPU="${GPU:-H100_SXM}"
PTZ="${PTZ:-final_model.int6.ptz}"
MIN_VRAM=80000          # 80GB
MIN_RELIABILITY=0.95
MAX_PRICE=2.50          # $/hr cap
DISK_GB=60
SSH_KEY="$HOME/.ssh/id_ed25519_apollo"
IMAGE="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${LOCAL_DIR}/results/vast_sweeps"
POLL_INTERVAL=10        # seconds between status checks
MAX_WAIT=300            # max seconds to wait for instance ready

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --grid)  GRID="$2"; shift 2 ;;
        --gpu)   GPU="$2"; shift 2 ;;
        --ptz)   PTZ="$2"; shift 2 ;;
        --price) MAX_PRICE="$2"; shift 2 ;;
        *)       echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LABEL="ttt_${GRID}_${TIMESTAMP}"

echo "============================================"
echo "  Vast.ai TTT Sweep"
echo "  Grid:  $GRID"
echo "  GPU:   $GPU"
echo "  PTZ:   $PTZ"
echo "  Label: $RUN_LABEL"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Preflight checks
# ---------------------------------------------------------------------------
if ! command -v vastai &>/dev/null; then
    echo "ERROR: vastai CLI not installed. Run: pip install vastai"
    exit 1
fi

if [ ! -f "$HOME/.vast_api_key" ]; then
    echo "ERROR: No Vast.ai API key. Run: vastai set api-key YOUR_KEY"
    exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
    echo "ERROR: SSH key not found at $SSH_KEY"
    exit 1
fi

# Check that the .ptz file exists (local or in checkpoints/)
PTZ_PATH=""
for candidate in "$PTZ" "${LOCAL_DIR}/${PTZ}" "${LOCAL_DIR}/checkpoints/${PTZ}"; do
    if [ -f "$candidate" ]; then
        PTZ_PATH="$candidate"
        break
    fi
done
if [ -z "$PTZ_PATH" ]; then
    echo "ERROR: Cannot find .ptz file: $PTZ"
    echo "  Searched: $PTZ, ${LOCAL_DIR}/${PTZ}, ${LOCAL_DIR}/checkpoints/${PTZ}"
    exit 1
fi
echo "==> PTZ file: $PTZ_PATH ($(ls -lh "$PTZ_PATH" | awk '{print $5}'))"

# ---------------------------------------------------------------------------
# Step 1: Find cheapest matching GPU
# ---------------------------------------------------------------------------
echo ""
echo "==> Searching for ${GPU} instances (max \$${MAX_PRICE}/hr)..."

OFFER_JSON=$(vastai search offers \
    "gpu_name=${GPU} num_gpus=1 gpu_ram>=${MIN_VRAM} reliability>${MIN_RELIABILITY} rentable=True dph_total<=${MAX_PRICE} verified=True" \
    -t on-demand -o 'dph_total' --raw 2>/dev/null | head -1)

if [ -z "$OFFER_JSON" ] || [ "$OFFER_JSON" = "[]" ]; then
    echo "No ${GPU} offers found under \$${MAX_PRICE}/hr. Trying A100_SXM..."
    GPU="A100_SXM"
    OFFER_JSON=$(vastai search offers \
        "gpu_name=${GPU} num_gpus=1 gpu_ram>=${MIN_VRAM} reliability>${MIN_RELIABILITY} rentable=True dph_total<=${MAX_PRICE} verified=True" \
        -t on-demand -o 'dph_total' --raw 2>/dev/null | head -1)
fi

if [ -z "$OFFER_JSON" ] || [ "$OFFER_JSON" = "[]" ]; then
    echo "ERROR: No offers found. Try increasing --price or check vast.ai"
    exit 1
fi

# Parse offer — vastai --raw returns a JSON array, grab first (cheapest) entry
OFFER_ID=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'] if isinstance(d,list) else d['id'])" 2>/dev/null || echo "")
OFFER_PRICE=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); e=d[0] if isinstance(d,list) else d; print(f\"{e['dph_total']:.3f}\")" 2>/dev/null || echo "?")
OFFER_GPU=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); e=d[0] if isinstance(d,list) else d; print(e.get('gpu_name','?'))" 2>/dev/null || echo "?")

if [ -z "$OFFER_ID" ]; then
    echo "ERROR: Could not parse offer. Raw response:"
    echo "$OFFER_JSON" | head -5
    exit 1
fi

echo "==> Best offer: ID=${OFFER_ID}  GPU=${OFFER_GPU}  \$${OFFER_PRICE}/hr"
echo ""
read -p "Rent this instance? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2: Create instance
# ---------------------------------------------------------------------------
echo "==> Creating instance..."

CREATE_OUT=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh \
    --direct \
    --onstart-cmd "echo VAST_READY" \
    --label "$RUN_LABEL" \
    2>&1)

echo "$CREATE_OUT"
INSTANCE_ID=$(echo "$CREATE_OUT" | grep -oE 'new_contract["\s:]+[0-9]+' | grep -oE '[0-9]+' | head -1)
if [ -z "$INSTANCE_ID" ]; then
    # Try alternate parse
    INSTANCE_ID=$(echo "$CREATE_OUT" | grep -oE 'instance [0-9]+|ID: [0-9]+|"id":\s*[0-9]+' | grep -oE '[0-9]+' | head -1)
fi

if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: Could not parse instance ID from create response"
    exit 1
fi

echo "==> Instance ID: $INSTANCE_ID"

# ---------------------------------------------------------------------------
# Step 3: Wait for instance to be ready
# ---------------------------------------------------------------------------
echo "==> Waiting for instance to start..."
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "running" ]; then
        echo "==> Instance is running!"
        break
    fi
    echo "    status=$STATUS  (${WAITED}s / ${MAX_WAIT}s)"
    sleep $POLL_INTERVAL
    WAITED=$((WAITED + POLL_INTERVAL))
done

if [ "$STATUS" != "running" ]; then
    echo "ERROR: Instance did not start within ${MAX_WAIT}s. Destroying..."
    vastai destroy instance "$INSTANCE_ID"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 4: Get SSH connection details
# ---------------------------------------------------------------------------
echo "==> Getting SSH connection info..."
sleep 5  # brief pause for SSH to be ready

SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
# Parse: ssh -p PORT root@HOST
SSH_PORT=$(echo "$SSH_URL" | grep -oE '\-p [0-9]+' | awk '{print $2}')
SSH_HOST=$(echo "$SSH_URL" | grep -oE '[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+' | tail -1)

if [ -z "$SSH_PORT" ] || [ -z "$SSH_HOST" ]; then
    echo "ERROR: Could not parse SSH URL: $SSH_URL"
    echo "Destroying instance..."
    vastai destroy instance "$INSTANCE_ID"
    exit 1
fi

echo "==> SSH: $SSH_HOST port $SSH_PORT"

SSH_CMD="ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i $SSH_KEY -p $SSH_PORT $SSH_HOST"
SCP_CMD="scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i $SSH_KEY -P $SSH_PORT"

# Test SSH
echo "==> Testing SSH connection..."
RETRIES=0
while [ $RETRIES -lt 6 ]; do
    if $SSH_CMD "echo HELLO_VAST" 2>/dev/null | grep -q HELLO_VAST; then
        echo "    SSH OK"
        break
    fi
    RETRIES=$((RETRIES + 1))
    echo "    retry $RETRIES..."
    sleep 5
done

if [ $RETRIES -ge 6 ]; then
    echo "ERROR: SSH connection failed after retries"
    vastai destroy instance "$INSTANCE_ID"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 5: Upload payload
# ---------------------------------------------------------------------------
echo "==> Preparing payload..."

PAYLOAD_DIR=$(mktemp -d)
trap "rm -rf $PAYLOAD_DIR" EXIT

# Copy only what's needed
cp "$LOCAL_DIR/train_gpt_v7_submit.py" "$PAYLOAD_DIR/"
cp "$LOCAL_DIR/sweep_ttt_single_gpu.py" "$PAYLOAD_DIR/"
cp "$PTZ_PATH" "$PAYLOAD_DIR/model.ptz"

# Data: val shard + tokenizer
mkdir -p "$PAYLOAD_DIR/data/datasets/fineweb10B_sp1024"
mkdir -p "$PAYLOAD_DIR/data/tokenizers"
cp "$LOCAL_DIR/data/datasets/fineweb10B_sp1024/fineweb_val_"*.bin "$PAYLOAD_DIR/data/datasets/fineweb10B_sp1024/" 2>/dev/null || true
cp "$LOCAL_DIR/data/tokenizers/fineweb_1024_bpe.model" "$PAYLOAD_DIR/data/tokenizers/" 2>/dev/null || true

# Create tarball
TARBALL="/tmp/vast_payload_${TIMESTAMP}.tar.gz"
(cd "$PAYLOAD_DIR" && tar czf "$TARBALL" .)
echo "==> Payload: $(ls -lh "$TARBALL" | awk '{print $5}')"

echo "==> Uploading payload..."
$SCP_CMD "$TARBALL" "${SSH_HOST}:/workspace/payload.tar.gz"

echo "==> Extracting on instance..."
$SSH_CMD "cd /workspace && tar xzf payload.tar.gz && ls -lh && echo EXTRACT_OK" 2>/dev/null

# Install deps — flash-attn v2 provides flash_attn_func with same (q,k,v,causal) signature
# Create a shim so `from flash_attn_interface import flash_attn_func` works everywhere
echo "==> Installing dependencies..."
$SSH_CMD "pip install sentencepiece zstandard flash-attn --no-build-isolation 2>&1 | tail -5" 2>/dev/null
$SSH_CMD "python3 -c \"
import os, sys
# Create flash_attn_interface shim that maps to flash_attn v2
shim = '''
try:
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    from torch.nn.functional import scaled_dot_product_attention as _sdpa
    import torch
    def flash_attn_func(q, k, v, causal=False):
        # q,k,v: (B, T, H, D) -> SDPA expects (B, H, T, D)
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

# ---------------------------------------------------------------------------
# Step 6: Run the sweep
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Running TTT sweep: --grid $GRID"
echo "  Instance: $INSTANCE_ID ($OFFER_GPU @ \$$OFFER_PRICE/hr)"
echo "============================================"
echo ""

$SSH_CMD "cd /workspace && python sweep_ttt_single_gpu.py \
    --ptz model.ptz \
    --grid $GRID \
    --output sweep_results_${RUN_LABEL}.json \
    2>&1" | tee "/tmp/vast_sweep_${RUN_LABEL}.log"

# ---------------------------------------------------------------------------
# Step 7: Pull results
# ---------------------------------------------------------------------------
echo ""
echo "==> Pulling results..."
mkdir -p "$RESULTS_DIR"

$SCP_CMD "${SSH_HOST}:/workspace/sweep_results_${RUN_LABEL}.json" "$RESULTS_DIR/" 2>/dev/null || true

echo "==> Results saved to: $RESULTS_DIR/sweep_results_${RUN_LABEL}.json"

# Also save the log
cp "/tmp/vast_sweep_${RUN_LABEL}.log" "$RESULTS_DIR/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 8: Destroy instance
# ---------------------------------------------------------------------------
echo ""
echo "==> Destroying instance $INSTANCE_ID..."
vastai destroy instance "$INSTANCE_ID"
echo "==> Instance destroyed. No further charges."

echo ""
echo "============================================"
echo "  DONE"
echo "  Results: $RESULTS_DIR/sweep_results_${RUN_LABEL}.json"
echo "  Log:     $RESULTS_DIR/vast_sweep_${RUN_LABEL}.log"
echo "============================================"
