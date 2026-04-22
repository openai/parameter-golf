#!/bin/bash
# Quick throughput calibration — run on a fresh pod before committing to a full screen.
# Kills training after step 500 (first logged step post-warmup) and reports tok/s.
# Usage: bash tput_calibration.sh <pod_host> <pod_port> <data_dir> <commit>
#
# Decision threshold:
#   >= 6,400k tok/s @ step 500  → fast node (good)
#   6,200–6,400k               → acceptable (≤3% deficit)
#   < 6,200k                   → slow node — stop and reshop
#   < 5,500k                   → NA-tier — do not use for screens

set -euo pipefail

POD_HOST="${1:-}"
POD_PORT="${2:-}"
DATA_DIR="${3:-/workspace/data}"
COMMIT="${4:-4dd2d63}"
SSH_KEY="${SSH_KEY:-/home/claude-user/.runpod/ssh/RunPod-Key-Go}"

if [ -z "$POD_HOST" ] || [ -z "$POD_PORT" ]; then
  echo "Usage: $0 <pod_host> <pod_port> [data_dir] [commit]"
  exit 1
fi

JPSSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$POD_HOST -p $POD_PORT"

echo "=== Throughput calibration: $POD_HOST:$POD_PORT commit=$COMMIT ==="

# Clone repo if needed
$JPSSH "bash -s" 2>/dev/null << ENDSSH
set -e
if [ ! -d /runpod/tput_test ] && [ ! -d /workspace/tput_test ]; then
  DEST=\$([ -d /runpod ] && echo /runpod || echo /workspace)
  git clone https://github.com/leon2k2k2k/parameter-golf.git \$DEST/tput_test
fi
DEST=\$([ -d /runpod/tput_test ] && echo /runpod || echo /workspace)
cd \$DEST/tput_test
git fetch origin
git checkout $COMMIT

pip install brotli pyminify --break-system-packages -q 2>&1 | tail -2

CACHE=\$([ -d /runpod ] && echo /runpod/.torch_inductor_cache || echo /workspace/.torch_inductor_cache)
LOGDIR=\$([ -d /runpod ] && echo /runpod || echo /workspace)/runs/tput-calibration
mkdir -p \$LOGDIR \$CACHE

nohup bash -c "
NCCL_NET=Socket DATA_DIR=$DATA_DIR \\
TORCHINDUCTOR_CACHE_DIR=\$CACHE \\
CASEOPS_ENABLED=1 \\
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \\
RECUR_ALPHA_ENABLED=1 \\
SEED=42 \\
torchrun --standalone --nproc_per_node=8 \\
  \$DEST/tput_test/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \\
  > \$LOGDIR/train.log 2>&1
echo \$! > \$LOGDIR/train.pid
" &
sleep 2
echo "Training launched"
ENDSSH

echo "Polling for step 500 (first stable tok/s)..."
LOGDIR=$(${JPSSH} "[ -d /runpod ] && echo /runpod/runs/tput-calibration || echo /workspace/runs/tput-calibration" 2>/dev/null)

for i in $(seq 1 40); do
  LINES=$(${JPSSH} "grep 'tok/s' ${LOGDIR}/train.log 2>/dev/null | tail -5" 2>/dev/null | grep -v "Permanently")
  STEP=$(echo "$LINES" | grep -oE "^[0-9]+" | tail -1)
  TOKS=$(echo "$LINES" | tail -1 | grep -oE "tok/s: [0-9]+" | awk '{print $2}')
  PHASE=$(${JPSSH} "tail -1 ${LOGDIR}/train.log 2>/dev/null" 2>/dev/null | grep -v "Permanently" | cut -c1-80)

  if [ -n "$STEP" ] && [ "$STEP" -ge 500 ] 2>/dev/null; then
    echo ""
    echo "=== Step $STEP | tok/s: $TOKS ==="
    echo "$LINES"
    echo ""
    if   [ "$TOKS" -ge 6400000 ] 2>/dev/null; then echo "VERDICT: FAST NODE ✓ (>= 6.4M)";
    elif [ "$TOKS" -ge 6200000 ] 2>/dev/null; then echo "VERDICT: ACCEPTABLE (6.2-6.4M, ≤3% deficit)";
    elif [ "$TOKS" -ge 5500000 ] 2>/dev/null; then echo "VERDICT: SLOW NODE ✗ — reshop";
    else echo "VERDICT: NA-TIER ✗✗ — do not use"; fi
    # Kill training
    ${JPSSH} "pkill -9 -f torchrun 2>/dev/null; pkill -9 -f train_gpt 2>/dev/null; echo killed" 2>/dev/null | grep -v Permanently
    break
  fi
  printf "[%s] %s\n" "$(date +%H:%M:%S)" "${PHASE:-waiting...}"
  sleep 30
done
