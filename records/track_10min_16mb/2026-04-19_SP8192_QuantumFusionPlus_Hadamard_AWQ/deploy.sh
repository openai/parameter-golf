#!/bin/bash

# Complete RunPod Deployment Script
# One-command execution for full training pipeline

set -e

echo "=========================================="
echo "SP8192 + Quantum Fusion Plus"
echo "Complete RunPod Deployment"
echo "=========================================="
echo ""

# Get GitHub token from environment or user input
if [ -z "$GITHUB_TOKEN" ]; then
    echo "GitHub token not found in environment."
    echo "Please provide your GitHub Personal Access Token:"
    read -s GITHUB_TOKEN
    export GITHUB_TOKEN=$GITHUB_TOKEN
fi

# Step 1: Setup
echo "[Step 1/5] Setting up RunPod environment..."
bash setup_runpod.sh
echo ""

# Step 2: Download data
echo "[Step 2/5] Downloading Fineweb SP8192 dataset..."
bash download_data.sh
echo ""

# Step 3: Train
echo "[Step 3/5] Running 3-seed training..."
bash train_all_seeds.sh
echo ""

# Step 4: Collect results
echo "[Step 4/5] Collecting training results..."
bash collect_results.sh
echo ""

# Step 5: Upload
echo "[Step 5/5] Uploading results to GitHub..."
bash upload_to_github.sh
echo ""

echo "=========================================="
echo "✅ Deployment complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✅ Environment setup"
echo "  ✅ Data downloaded"
echo "  ✅ 3-seed training completed"
echo "  ✅ Results collected"
echo "  ✅ Results uploaded to GitHub"
echo ""
echo "PR #1732 has been updated with real training results!"
echo "Visit: https://github.com/openai/parameter-golf/pull/1732"
