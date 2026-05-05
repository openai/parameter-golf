#!/bin/bash
# Full environment setup for one-command reproduction.
# Tested on RunPod PyTorch 2.9.1+cu128 image. Adapt apt commands for non-Debian hosts.
# Usage: bash setup.sh
set -e

echo "=== [1/5] System packages (gcc + lrzip) ==="
NEED_APT=()
command -v gcc   >/dev/null 2>&1 || NEED_APT+=(build-essential)
command -v lrzip >/dev/null 2>&1 || NEED_APT+=(lrzip)
if [ ${#NEED_APT[@]} -gt 0 ]; then
    apt-get update -qq && apt-get install -y -qq "${NEED_APT[@]}"
fi
gcc  --version | head -1
lrzip -V 2>&1 | head -1

echo "=== [2/5] PyTorch 2.9.1 + Triton ==="
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "0.0.0")
if echo "$TORCH_VER" | grep -q "2.9"; then
    echo "  PyTorch $TORCH_VER OK"
else
    pip install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
fi
python3 -c "import triton; print(f'  Triton {triton.__version__} OK')"

echo "=== [3/5] Python deps + hf CLI ==="
pip install -q -U \
    numpy tqdm "huggingface-hub[cli]>=0.27" datasets tiktoken sentencepiece kernels \
    "typing-extensions==4.15.0" zstandard brotli
hash -r
# hf CLI is the modern Hugging Face command-line tool (replaces legacy huggingface-cli)
if command -v hf >/dev/null 2>&1; then
    echo "  hf CLI: $(hf --version 2>&1 | head -1)"
else
    echo "  hf CLI MISSING — install failed"; exit 1
fi

echo "=== [4/5] Flash Attention 3 ==="
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null && echo "  FlashAttn3 OK" || {
    pip install -q flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 || \
        pip install -q flash-attn --no-build-isolation
}

echo "=== [5/5] CASEOPS data preparation ==="
DATA_DIR="${DATA_DIR:-/runpod-volume/caseops_data/datasets}"
DATA_PATH="${DATA_PATH:-$DATA_DIR/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}"
SIDECARS=$(ls "$DATA_PATH"/fineweb_val_bytes_*.bin 2>/dev/null | wc -l)

if [ "$SIDECARS" -ge 1 ]; then
    echo "  CASEOPS data already present ($SIDECARS val sidecars at $DATA_PATH)"
else
    echo "  CASEOPS data missing — preparing from raw FineWeb shards..."
    DOCS_JSONL="${DOCS_JSONL:-/runpod-volume/hf_cache/docs_selected.jsonl}"
    if [ ! -f "$DOCS_JSONL" ]; then
        echo "  Downloading raw docs_selected.jsonl via hf CLI..."
        mkdir -p "$(dirname "$DOCS_JSONL")"
        # hf download <repo_id> <filename> --repo-type dataset --local-dir <dir>
        hf download "${MATCHED_FINEWEB_REPO_ID:-willdepueoai/parameter-golf}" \
            datasets/docs_selected.jsonl \
            --repo-type dataset \
            --local-dir "$(dirname "$DOCS_JSONL")"
        # hf download places file at <local-dir>/datasets/docs_selected.jsonl;
        # symlink to expected flat path if needed.
        NESTED="$(dirname "$DOCS_JSONL")/datasets/docs_selected.jsonl"
        if [ -f "$NESTED" ] && [ ! -f "$DOCS_JSONL" ]; then
            ln -s "$NESTED" "$DOCS_JSONL"
        fi
    fi
    mkdir -p "$DATA_PATH" "$DATA_DIR/tokenizers"
    cp -n "$(dirname "$0")/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
        "$DATA_DIR/tokenizers/" 2>/dev/null || true
    echo "  Tokenizing with CASEOPS SP8192 model (CPU, ~10-20 min)..."
    python3 "$(dirname "$0")/prepare_caseops_data.py" \
        --docs "$DOCS_JSONL" \
        --out "$DATA_DIR" \
        --sp "$(dirname "$0")/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
    SIDECARS=$(ls "$DATA_PATH"/fineweb_val_bytes_*.bin 2>/dev/null | wc -l)
    if [ "$SIDECARS" -lt 1 ]; then
        echo "  ERROR: CASEOPS prep failed — no val sidecars at $DATA_PATH"
        exit 1
    fi
    echo "  CASEOPS prep done ($SIDECARS val sidecars)"
fi

echo ""
echo "=== Environment ready ==="
python3 -c "
import torch, triton
print(f'  PyTorch  {torch.__version__}')
print(f'  Triton   {triton.__version__}')
print(f'  CUDA     {torch.version.cuda}')
print(f'  GPUs:    {torch.cuda.device_count()}')
try:
    from flash_attn_interface import flash_attn_func
    print('  FlashAttn3: OK')
except Exception:
    print('  FlashAttn3: MISSING')
"
echo ""
echo "Next: SEED=42 bash run.sh    (then SEED=0 and SEED=1234 for the other declared seeds)"
