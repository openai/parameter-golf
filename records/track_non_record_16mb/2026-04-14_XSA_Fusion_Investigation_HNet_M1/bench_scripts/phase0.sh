#!/usr/bin/env bash
# Parameter Golf — Phase 0: env sanity + clone + deps + data + fork baseline.
# Run on the RunPod 1xH100 pod (pytorch:1.0.2-cu1281-torch280-ubuntu2404).
# Usage:  bash phase0.sh 2>&1 | tee phase0.log

set -euo pipefail
cd /workspace
echo "=== PHASE 0 START ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

# 1. env sanity -------------------------------------------------------------
echo "--- env ---"
python - <<'PY'
import sys, torch, triton
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
print("sm", torch.cuda.get_device_capability(0))
PY
df -h /workspace | tail -1

# 2. clone repo -------------------------------------------------------------
echo "--- clone ---"
if [ ! -d parameter-golf ]; then
  git clone --depth 1 https://github.com/openai/parameter-golf.git
fi
cd parameter-golf
git log -1 --oneline

# 3. install deps -----------------------------------------------------------
echo "--- pip ---"
pip install -q --no-input brotli sentencepiece huggingface-hub datasets tqdm 2>&1 | tail -3
python -c "import brotli, sentencepiece, huggingface_hub; print('brotli', brotli.__version__); print('sp', sentencepiece.__version__); print('hf_hub', huggingface_hub.__version__)"

# 4. download SP8192 smoke subset (2 train shards + full val) ---------------
echo "--- data ---"
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 2 2>&1 | tail -15
ls -lh data/datasets/fineweb10B_sp8192/ | head -8
ls -lh data/tokenizers/ | head -8

# 5. fork bigbag's top record as our working baseline -----------------------
echo "--- fork baseline ---"
mkdir -p /workspace/work
cp -v records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py \
      /workspace/work/train_gpt_baseline.py
wc -l /workspace/work/train_gpt_baseline.py
md5sum /workspace/work/train_gpt_baseline.py

echo "=== PHASE 0 DONE ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
