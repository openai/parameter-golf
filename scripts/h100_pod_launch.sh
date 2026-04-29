#!/usr/bin/env bash
# h100_pod_launch.sh — Full 8xH100 pod setup + 3-seed H100 final run launcher.
#
# Run this from your LOCAL machine after setting the pod SSH address.
#
# Usage (direct TCP — recommended, supports remote commands):
#   export RP_HOST="root@<ip>"         # e.g. root@103.207.149.64
#   export RP_PORT="<port>"            # e.g. 10679  (from pod Connect > SSH over TCP)
#   export RP_KEY="/path/to/ssh_key"
#   bash scripts/h100_pod_launch.sh [probe|setup|run|status|logs|results]

set -euo pipefail

RP_HOST="${RP_HOST:?Set RP_HOST e.g. root@<ip>}"
RP_PORT="${RP_PORT:-22}"
RP_KEY="${RP_KEY:?Set RP_KEY to your SSH private key path}"
RP_REPO="/workspace/parameter-golf-main"
FORK_URL="https://github.com/harborglowvintage-oss/parameter-golf.git"
BATCH_LOG="/workspace/h100_batch.log"

SSH_OPTS="-i ${RP_KEY} -p ${RP_PORT} -o IdentitiesOnly=yes -o IdentityAgent=none -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=5 -o NumberOfPasswordPrompts=0"

ssh_cmd() {
  # $1 = remote bash command (single string)
  timeout 1200 ssh $SSH_OPTS "${RP_HOST}" "$1"
}

PHASE="${1:-setup}"

if [[ "$PHASE" == "probe" ]]; then
  echo "=== PROBE ==="
  timeout 20 ssh $SSH_OPTS "${RP_HOST}" "hostname; date -u; nvidia-smi -L; echo PROBE_OK"
  exit 0
fi

if [[ "$PHASE" == "setup" ]]; then
  echo "=== PHASE 1: Clone repo ==="
  ssh_cmd "
    set -euo pipefail
    cd /workspace
    rm -rf parameter-golf-main
    git clone '${FORK_URL}' parameter-golf-main
    cd parameter-golf-main
    python3 -c \"import ast; ast.parse(open('train_gpt_sota_decoded.py').read()); print('SYNTAX_OK')\"
  "

  echo "=== PHASE 2: Install dependencies (includes flash-attn pre-built wheel ~2 min) ==="
  ssh_cmd "
    set -euo pipefail
    cd '${RP_REPO}'
    pip install -r requirements.txt -q
    python3 -c "import flash_attn; print('FLASH_ATTN_OK', flash_attn.__version__)"
  "

  echo "=== PHASE 3: Download training data (fineweb10B_sp8192, ~15-30 min) ==="
  timeout 2400 ssh $SSH_OPTS "${RP_HOST}" "
    set -euo pipefail
    cd '${RP_REPO}'
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
    ls data/datasets/fineweb10B_sp8192/*.bin | wc -l
    ls data/tokenizers/fineweb_8192_bpe.model && echo 'TOKENIZER_OK'
  "

  echo "=== SETUP COMPLETE. Now run: bash scripts/h100_pod_launch.sh run ==="
  exit 0
fi

if [[ "$PHASE" == "run" ]]; then
  echo "=== Launching 3-seed H100 batch ==="
  ssh_cmd "
    set -euo pipefail
    cd '${RP_REPO}'
    mkdir -p logs/sweep

    # Quick sanity check before launching
    python3 -c \"
import flash_attn; print('flash_attn', flash_attn.__version__)
import torch; print('torch', torch.__version__, 'GPUs:', torch.cuda.device_count())
import ast; ast.parse(open('train_gpt_sota_decoded.py').read()); print('SYNTAX_OK')
\"

    nohup bash -c '
      set -u
      cd ${RP_REPO}
      echo \"=== SEED 42 START \$(date -u) ===\"
      bash scripts/sweep_runner.sh scripts/sweeps/h100_mlr015_e7_s42.tsv
      echo \"=== SEED 314 START \$(date -u) ===\"
      bash scripts/sweep_runner.sh scripts/sweeps/h100_mlr015_e7_s314.tsv
      echo \"=== SEED 1337 START \$(date -u) ===\"
      bash scripts/sweep_runner.sh scripts/sweeps/h100_mlr015_e7_s1337.tsv
      echo \"=== ALL_DONE \$(date -u) ===\"
    ' > '${BATCH_LOG}' 2>&1 &
    echo MASTER_PID=\$!
    echo BATCH_LOG=${BATCH_LOG}
  "
  exit 0
fi

if [[ "$PHASE" == "status" ]]; then
  echo "=== STATUS ==="
  timeout 30 ssh $SSH_OPTS "${RP_HOST}" "
    echo '--- Processes ---'
    ps -ef | grep -E 'sweep_runner|run_experiment|torchrun|train_gpt' | grep -v grep || echo IDLE
    echo '--- Last 40 lines of batch log ---'
    tail -n 40 '${BATCH_LOG}' 2>/dev/null || echo 'No batch log yet'
    echo '--- Results CSV ---'
    tail -n 10 '${RP_REPO}/logs/sweep/results.csv' 2>/dev/null || echo 'No results yet'
  "
  exit 0
fi

if [[ "$PHASE" == "logs" ]]; then
  echo "=== Streaming batch log (ctrl-c to stop) ==="
  timeout 7200 ssh $SSH_OPTS "${RP_HOST}" "tail -f '${BATCH_LOG}'" || true
  exit 0
fi

if [[ "$PHASE" == "results" ]]; then
  echo "=== Results ==="
  timeout 30 ssh $SSH_OPTS "${RP_HOST}" "
    cat '${RP_REPO}/logs/sweep/results.csv' 2>/dev/null || echo 'No results yet'
    echo '--- Artifacts ---'
    ls -la '${RP_REPO}'/final_model*.pt* 2>/dev/null || echo 'No artifacts yet'
  "
  exit 0
fi

echo "Usage: $0 [probe|setup|run|status|logs|results]"
exit 1
