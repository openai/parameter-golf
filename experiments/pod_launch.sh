#!/bin/bash
set -euo pipefail
# POD LAUNCH — one command to rule them all
# Usage: curl -sL <raw_url> | bash -s [experiment_script]
#   or:  bash experiments/pod_launch.sh experiments/A_wing/purple/run.sh
#
# Handles: git clone/checkout, env setup, then runs your experiment.

REPO_URL="https://github.com/newjordan/parameter-golf-1.git"
BRANCH="${BRANCH:-test}"
WORKSPACE="/workspace/parameter-golf-lab"
REMOTE_NAME="fork1"
EXPERIMENT="${1:-}"

echo "============================================"
echo "  POD LAUNCH — Auto Setup + Run"
echo "  Branch: ${BRANCH}"
echo "  Experiment: ${EXPERIMENT:-<none, setup only>}"
echo "============================================"

# --- Step 1: Get the repo ---
if [ -d "${WORKSPACE}/.git" ]; then
    echo "[1/3] Repo exists, force-syncing to ${BRANCH}..."
    cd "${WORKSPACE}"
    # Ensure private remote exists
    git remote get-url "${REMOTE_NAME}" &>/dev/null || git remote add "${REMOTE_NAME}" "${REPO_URL}"
    git fetch "${REMOTE_NAME}" "${BRANCH}" --quiet
    git checkout -B "${BRANCH}" "${REMOTE_NAME}/${BRANCH}" --force
    git clean -fd --quiet
else
    echo "[1/3] Cloning repo..."
    git clone -b "${BRANCH}" "${REPO_URL}" "${WORKSPACE}"
    cd "${WORKSPACE}"
fi
echo "  HEAD: $(git log --oneline -1)"

# --- Step 2: Environment setup ---
echo "[2/3] Running setup_runpod.sh..."
bash experiments/setup_runpod.sh

# --- Step 3: Run experiment ---
if [ -n "${EXPERIMENT}" ]; then
    echo "[3/3] Launching: ${EXPERIMENT}"
    bash "${EXPERIMENT}"
else
    echo "[3/3] No experiment specified. Ready to run manually."
    echo "  Example: bash experiments/A_wing/purple/run.sh"
fi
