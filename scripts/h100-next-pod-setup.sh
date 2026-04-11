#!/bin/bash
# =============================================================================
# NEXT POD SETUP SCRIPT
# Phase A: SP4096 + MLP4x + full architecture stack
# Updated: 2026-04-05
# =============================================================================
# Usage: bash h100-next-pod-setup.sh
# This sets up the environment, downloads SP4096 data, then launches runs.
# =============================================================================

set -e
cd /workspace

# ---- 0. Benchmark this pod first ----
echo "=== POD BENCHMARK ==="
curl -s https://raw.githubusercontent.com/NathanMaine/runpod-gpu-benchmark/main/pod-test.sh | bash || true
echo "==========================="
echo "If GEMM > 0.70ms or MemBW < 2000 GB/s → SWITCH PODS before continuing!"
read -p "Pod OK? (y/n): " pod_ok
if [[ "$pod_ok" != "y" ]]; then
  echo "Aborting — switch pods first."
  exit 1
fi

# ---- 1. Clone repo ----
echo "=== Cloning repo ==="
if [ ! -d "parameter-golf" ]; then
  git clone https://github.com/Programmerryoki/parameter-golf.git
fi
cd parameter-golf

# ---- 2. Install deps ----
echo "=== Installing dependencies ==="
pip install -q brotli zstandard tiktoken 2>&1 | tail -5

# ---- 3. Download SP4096 data ----
echo "=== Downloading SP4096 dataset ==="
mkdir -p data/datasets
if [ ! -d "data/datasets/fineweb10B_sp4096" ] || [ "$(ls data/datasets/fineweb10B_sp4096/*.bin 2>/dev/null | wc -l)" -lt 80 ]; then
  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
    python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest
  echo "SP4096 data downloaded."
else
  echo "SP4096 data already present ($(ls data/datasets/fineweb10B_sp4096/*.bin | wc -l) shards)."
fi

# ---- 4. Download SP1024 data (backup/baseline) ----
echo "=== Checking SP1024 dataset ==="
if [ ! -d "data/datasets/fineweb10B_sp1024" ] || [ "$(ls data/datasets/fineweb10B_sp1024/*.bin 2>/dev/null | wc -l)" -lt 80 ]; then
  echo "Downloading SP1024..."
  MATCHED_FINEWEB_REPO_ID=willdepueoai/parameter-golf \
    python3 data/cached_challenge_fineweb.py --variant sp1024 --skip-manifest || true
fi

echo ""
echo "=== SETUP COMPLETE ==="
echo "Now run: bash scripts/h100-sp4096-ablations.sh"
echo ""
