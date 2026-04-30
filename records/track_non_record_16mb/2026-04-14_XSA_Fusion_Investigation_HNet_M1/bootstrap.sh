#!/usr/bin/env bash
# Bootstrap a FRESH RunPod 1xH100 pod (pytorch:1.0.2-cu1281-torch280-ubuntu2404)
# from scratch to the state where phase4.sh (H-Net M1 pilot) can run.
#
# Upload this file + unpack.py + hnet_m1/ to /workspace/ before running.
#
# Expected cost: ~5 min wallclock on 1xH100 = ~$0.30.
#
# Usage:  bash bootstrap.sh 2>&1 | tee bootstrap.log
set -euo pipefail
cd /workspace

echo "=== BOOTSTRAP ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

# 1. env sanity --------------------------------------------------------------
echo "--- env ---"
python - <<'PY'
import sys, torch, triton
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("gpu", torch.cuda.get_device_name(0), "sm", torch.cuda.get_device_capability(0))
PY

# 2. clone repo --------------------------------------------------------------
echo "--- clone parameter-golf ---"
if [ ! -d parameter-golf ]; then
    git clone --depth 1 https://github.com/openai/parameter-golf.git
fi
cd parameter-golf
git log -1 --oneline

# 3. python deps -------------------------------------------------------------
echo "--- pip deps ---"
pip install -q --no-input brotli sentencepiece huggingface-hub datasets tqdm 2>&1 | tail -3

# 4. SP8192 data (2 train shards smoke subset + full val) --------------------
echo "--- SP8192 data ---"
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
    python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 2 2>&1 | tail -5
ls -lh data/datasets/fineweb10B_sp8192/
ls -lh data/tokenizers/

# 5. unpack bigbag's baseline -----------------------------------------------
echo "--- unpack bigbag train_gpt.py ---"
mkdir -p /workspace/work
REC=records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT
python3 /workspace/unpack.py "$REC/train_gpt.py" /workspace/work/train_gpt_baseline.py
wc -l /workspace/work/train_gpt_baseline.py

# 6. FA3 install -------------------------------------------------------------
echo "--- FA3 install (cu128_torch280) ---"
pip install --quiet --no-deps flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280/ 2>&1 | tail -3
python -c "import flash_attn_interface as fa3; print('FA3 OK:', fa3.__file__)"

echo
echo "=== BOOTSTRAP DONE ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
echo
echo "Next: bash /workspace/hnet_m1/phase4.sh 2>&1 | tee /workspace/hnet_m1_pilot.log"
