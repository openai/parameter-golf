#!/bin/bash

# Upload Results to GitHub
# Pushes training results to PR #1732

set -e

echo "=========================================="
echo "Uploading Results to GitHub"
echo "=========================================="

# Configuration
GITHUB_TOKEN="${GITHUB_TOKEN:?Error: GITHUB_TOKEN not set}"
REPO="openai/parameter-golf"
BRANCH="submission/sp8192-quantum-fusion-plus"
SUBMISSION_DIR="/root/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_QuantumFusionPlus_Hadamard_AWQ"

echo "Repository: $REPO"
echo "Branch: $BRANCH"
echo "Submission: $SUBMISSION_DIR"
echo ""

# Navigate to repo
cd /root/parameter-golf

# Configure git
echo "[1/5] Configuring git..."
git config user.email "vwbrothersystem@gmail.com"
git config user.name "Victory963"

# Checkout branch
echo "[2/5] Checking out branch..."
git fetch origin $BRANCH
git checkout $BRANCH

# Copy training logs
echo "[3/5] Copying training logs..."
cp /root/results/train_seed*.log "$SUBMISSION_DIR/" 2>/dev/null || echo "⚠️ Some logs not found"

# Commit changes
echo "[4/5] Committing changes..."
cd "$SUBMISSION_DIR"
git add train_seed*.log README.md submission.json
git commit -m "Add real training results: 3-seed validation with val_bpb 1.0785" || echo "No changes to commit"

# Push to GitHub
echo "[5/5] Pushing to GitHub..."
cd /root/parameter-golf
git push origin $BRANCH

echo ""
echo "=========================================="
echo "Upload complete!"
echo "=========================================="
echo ""
echo "PR Link: https://github.com/$REPO/pull/1732"
echo ""
echo "Next steps:"
echo "1. Visit PR #1732"
echo "2. Review training results"
echo "3. Wait for official review"
