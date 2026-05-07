#!/bin/bash
set -e
# NOTE: Use a 200 GB volume disk on RunPod (mounted at /workspace).
# Data prep downloads ~48 GB of raw docs; they are deleted after tokenization,
# leaving ~20 GB for the CaseOps dataset shards.

SOTA_RECORD="records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611"
SOTA_RAW="https://raw.githubusercontent.com/openai/parameter-golf/main/${SOTA_RECORD}"

# ── 1. Upgrade PyTorch to cu128 (matches SOTA environment) ───────────────────
pip install -q --upgrade \
  torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# ── 2. Install Python deps ────────────────────────────────────────────────────
pip install -q -r requirements.txt

# ── 3. Install Flash Attention 3 (FA3) ───────────────────────────────────────
pip install flash_attn_3 --no-deps \
  --trusted-host windreamer.github.io \
  --find-links http://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')" \
  || { echo "ERROR: flash_attn_interface not available."; exit 1; }

# ── 4. Install lrzip (needed for per-group ZPAQ compression) ─────────────────
DEBIAN_FRONTEND=noninteractive apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y -q lrzip

# ── 5. Download SOTA record files (train_gpt.py + CaseOps helpers) ────────────
mkdir -p "${SOTA_RECORD}/tokenizers"
wget -q -O "${SOTA_RECORD}/train_gpt.py"            "${SOTA_RAW}/train_gpt.py"
wget -q -O "${SOTA_RECORD}/lossless_caps.py"         "${SOTA_RAW}/lossless_caps.py"
wget -q -O "${SOTA_RECORD}/prepare_caseops_data.py"  "${SOTA_RAW}/prepare_caseops_data.py"
wget -q -O "${SOTA_RECORD}/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
           "${SOTA_RAW}/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
echo "SOTA record files downloaded."

# ── 6. Build CaseOps dataset ──────────────────────────────────────────────────
# Route HF cache to /workspace so we can monitor & clean it up.
export HF_HOME=/workspace/.hf_cache

# Download docs_selected.jsonl (~35-40 GB) from the canonical dataset repo.
DOCS_JSONL=$(python3 - <<'EOF'
import sys
from huggingface_hub import hf_hub_download
for repo_id in ("willdepueoai/parameter-golf", "kevclark/parameter-golf"):
    try:
        path = hf_hub_download(
            repo_id, filename="docs_selected.jsonl",
            subfolder="datasets", repo_type="dataset"
        )
        print(path, end="")
        sys.exit(0)
    except Exception as e:
        print(f"  {repo_id}: {e}", file=sys.stderr)
print("ERROR: docs_selected.jsonl not found in any repo", file=sys.stderr)
sys.exit(1)
EOF
)
echo "docs_selected.jsonl: ${DOCS_JSONL}"

# Tokenize with CaseOps using parallel workers (~20 min vs 10 hours single-threaded).
# Strategy: stream docs_selected.jsonl, split across N workers, each writes its own
# temp train shards; first worker also handles all val docs (first 10k).
# Merges and renumbers shards at the end.
NWORKERS=$(nproc)
SP_MODEL="${SOTA_RECORD}/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
python3 experiments/prepare_caseops_parallel.py \
  --docs    "${DOCS_JSONL}" \
  --out     data \
  --sp      "${SP_MODEL}" \
  --workers "${NWORKERS}"

# Free ~40 GB used by docs_selected.jsonl HF cache — no longer needed.
rm -rf /workspace/.hf_cache
echo "HF cache cleared."

# ── 7. Install tokenizer where train_gpt.py expects it ───────────────────────
mkdir -p data/tokenizers
cp "${SOTA_RECORD}/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
   data/tokenizers/

echo "Setup complete."
echo "CaseOps dataset:"
du -sh data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/ 2>/dev/null || echo "  (not found — check prepare_caseops_data.py output above)"
df -h /workspace
