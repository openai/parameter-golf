#!/bin/bash
# ============================================================================
# Parameter Golf — RunPod Deployment Script
# ============================================================================
#
# COST ANALYSIS:
#   8xH100 SXM pod: ~$20/hr ($2.69/GPU × 8, billed per-second)
#   Training: 10 min = ~$3.33
#   Data download: ~5 min = ~$1.67
#   Setup + quantization + eval: ~5 min = ~$1.67
#   Total per run: ~$6.67
#   Budget for 5 runs: ~$35
#
# STRATEGY FOR LIMITED BUDGET:
#   1. Do ALL development on local 3070 (free)
#   2. Use 1xH100 ($2.69/hr) for quick validation if needed
#   3. Use 8xH100 ONLY for final submission runs
#   4. Pre-download data to a persistent volume (pay once, reuse)
#   5. Use the official template (no setup time wasted on dependencies)
#   6. Have the script ready to go — minimize idle pod time
#
# USAGE:
#   Phase 1 (one-time, ~$3): Create pod, download data to persistent volume
#     ./runpod_deploy.sh setup
#
#   Phase 2 (per-run, ~$3.33): Run training on 8xH100
#     ./runpod_deploy.sh train
#
#   Phase 3 (after training): Download results
#     ./runpod_deploy.sh results
#
# ============================================================================

set -e

# Configuration
SCRIPT_NAME="exp15_qkgain5_mlp4x.py"  # Our submission script
POD_NAME="parameter-golf-submission"
TEMPLATE_ID="y5cejece4j"
TEMPLATE_REF="nl2r56th"
GPU_TYPE="NVIDIA H100 SXM"
GPU_COUNT=8
VOLUME_SIZE=100  # GB, persistent across pod restarts
CONTAINER_DISK=20  # GB, ephemeral

# ============================================================================
# Phase 1: Setup (run once, ~5 min on pod = ~$1.67 for 8xH100)
# TIP: Do this on a 1xH100 pod ($2.69/hr) to save money, then stop it.
#       The persistent volume survives pod stops.
# ============================================================================
setup() {
    echo "============================================"
    echo "Phase 1: Setup — Download data to persistent volume"
    echo "============================================"
    echo ""
    echo "INSTRUCTIONS:"
    echo "1. Go to https://console.runpod.io/deploy?template=${TEMPLATE_ID}&ref=${TEMPLATE_REF}"
    echo "   OR create a pod manually:"
    echo "   - Template: Parameter Golf (ID: ${TEMPLATE_ID})"
    echo "   - GPU: 1x H100 SXM (for setup only — saves money)"
    echo "   - Volume: ${VOLUME_SIZE}GB persistent"
    echo ""
    echo "2. SSH into the pod and run:"
    echo ""
    cat << 'SETUP_COMMANDS'
# Clone repo and download ALL training data (this goes to persistent volume)
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Download full dataset (all shards, not just 10)
python3 data/cached_challenge_fineweb.py --variant sp1024
echo "Data download complete. $(ls data/datasets/fineweb10B_sp1024/*.bin | wc -l) shards."

# Install brotli for better compression
pip install brotli sentencepiece

# Verify GPU
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, Type: {torch.cuda.get_device_name(0)}')"

# STOP THE POD after data download to save money!
# The persistent volume keeps the data.
SETUP_COMMANDS
    echo ""
    echo "3. STOP the pod after data downloads. Don't delete — keep the volume."
    echo "   Cost: ~\$1-2 for data download on 1xH100."
}

# ============================================================================
# Phase 2: Train (run on 8xH100, exactly 10 min + overhead)
# ============================================================================
train() {
    echo "============================================"
    echo "Phase 2: Train — 8xH100, ~15 min total (~\$5)"
    echo "============================================"
    echo ""
    echo "INSTRUCTIONS:"
    echo "1. Change your pod to 8x H100 SXM (or create a new pod with the same volume)"
    echo "2. SSH in and run the following:"
    echo ""
    cat << 'TRAIN_COMMANDS'
cd /workspace/parameter-golf

# Upload our submission script (do this from your local machine first):
# scp experiments/r3/exp15_qkgain5_mlp4x.py root@POD_IP:/workspace/parameter-golf/train_submission.py
# OR paste it directly, OR git pull from your fork

# Ensure deps
pip install brotli sentencepiece 2>/dev/null

# Verify setup
echo "GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "Data shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"

# ============================================================
# THE SUBMISSION RUN
# ============================================================
# Key env vars:
#   MAX_WALLCLOCK_SECONDS=600  — 10-minute hard cap (competition rule)
#   ITERATIONS=20000           — will be capped by wallclock (~7000 steps)
#   WARMDOWN_ITERS=3500        — wallclock-aware, adjusts automatically
#   VAL_LOSS_EVERY=1000        — validate every 1000 steps
#   TRAIN_LOG_EVERY=100        — log every 100 steps

RUN_ID="submission_$(date +%Y%m%d_%H%M%S)" \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=20000 \
WARMDOWN_ITERS=3500 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=8 train_submission.py 2>&1 | tee "run_${RUN_ID}.log"

# ============================================================
# POST-TRAINING: Check results
# ============================================================
echo ""
echo "=== RESULTS ==="
grep "val_bpb" "run_${RUN_ID}.log" | grep "step:"
echo ""
grep "final_int6" "run_${RUN_ID}.log"
echo ""
grep "sliding_window" "run_${RUN_ID}.log"
echo ""
grep "Serialized" "run_${RUN_ID}.log"
echo ""
grep "size_prune" "run_${RUN_ID}.log"
echo ""
ls -la final_model.* 2>/dev/null

# Check if artifact fits
ARTIFACT_SIZE=$(stat -c%s final_model.int6.ptz 2>/dev/null || echo 0)
CODE_SIZE=$(wc -c < train_submission.py)
TOTAL=$((ARTIFACT_SIZE + CODE_SIZE))
echo ""
echo "Artifact: ${ARTIFACT_SIZE} bytes"
echo "Code: ${CODE_SIZE} bytes"
echo "Total: ${TOTAL} bytes"
if [ "$TOTAL" -le 16000000 ]; then
    echo "STATUS: FITS under 16MB ✓"
else
    echo "STATUS: OVER 16MB by $((TOTAL - 16000000)) bytes ✗"
fi
TRAIN_COMMANDS
    echo ""
    echo "3. After training: download results (Phase 3), then STOP the pod."
}

# ============================================================================
# Phase 3: Download results
# ============================================================================
results() {
    echo "============================================"
    echo "Phase 3: Download results"
    echo "============================================"
    echo ""
    echo "From your local machine:"
    echo ""
    cat << 'RESULTS_COMMANDS'
# Download the artifacts
POD_IP="YOUR_POD_IP"
scp root@${POD_IP}:/workspace/parameter-golf/final_model.int6.ptz ./
scp root@${POD_IP}:/workspace/parameter-golf/final_model.pt ./
scp root@${POD_IP}:/workspace/parameter-golf/run_*.log ./
scp root@${POD_IP}:/workspace/parameter-golf/train_submission.py ./

# STOP THE POD to stop billing!
RESULTS_COMMANDS
    echo ""
    echo "THEN: Stop the pod immediately to stop billing."
}

# ============================================================================
# Cost summary
# ============================================================================
cost() {
    echo "============================================"
    echo "Cost Estimate"
    echo "============================================"
    echo ""
    echo "Setup (one-time, 1xH100):        ~\$2   (5 min data download)"
    echo "Per submission run (8xH100):      ~\$5   (15 min: startup + 10 min train + 5 min eval)"
    echo "Volume storage (100GB):           ~\$7/month"
    echo ""
    echo "Budget for 5 submission runs:     ~\$27 + \$2 setup + \$7 storage = ~\$36"
    echo "Budget for 10 submission runs:    ~\$52 + \$2 setup + \$7 storage = ~\$61"
    echo ""
    echo "MONEY-SAVING TIPS:"
    echo "  1. Do ALL experimentation on your local RTX 3070 (free)"
    echo "  2. Use 1xH100 for quick validation runs (~\$2.69/hr, not \$20/hr)"
    echo "  3. Pre-download data once, reuse persistent volume"
    echo "  4. Have your script tested and ready BEFORE starting the 8xH100 pod"
    echo "  5. Stop the pod the SECOND training finishes — per-second billing"
    echo "  6. Apply for OpenAI's \$1M compute credits at the competition page"
    echo "  7. Don't use spot instances for submission runs — interruption = wasted money"
}

# ============================================================================
# Main
# ============================================================================
case "${1:-help}" in
    setup)   setup ;;
    train)   train ;;
    results) results ;;
    cost)    cost ;;
    help|*)
        echo "Usage: $0 {setup|train|results|cost}"
        echo ""
        echo "  setup    — Instructions to create pod and download data (one-time)"
        echo "  train    — Instructions to run training on 8xH100"
        echo "  results  — Instructions to download results"
        echo "  cost     — Cost breakdown and money-saving tips"
        ;;
esac
