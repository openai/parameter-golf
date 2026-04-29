#!/usr/bin/env bash
# nano_queue_s01.sh — Wait for data transfer to complete, then launch arch sweep on Nano.
# Run this ON SPARK — it SSHes into Nano to launch the sweep remotely.
#
# Usage:
#   nohup bash scripts/nano_queue_s01.sh > logs/sweep/nano_orchestrator_s01.log 2>&1 &
#   echo "nano_queue_s01 PID: $!"

set +e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

NANO_USER=ghostmini
NANO_HOST=192.168.179.119
NANO_KEY=~/.ssh/id_ed25519_nano
NANO_ROOT=/home/ghostmini
DATA_DIR=$NANO_ROOT/data
EXPECTED_SHARDS=101

log "=== nano_queue_s01 started — waiting for data transfer to Nano ==="

# Wait until all shards are present
while true; do
    count=$(ssh -i "$NANO_KEY" -o ConnectTimeout=10 "$NANO_USER@$NANO_HOST" \
        "ls \"$DATA_DIR/datasets/fineweb10B_sp8192/\"*.bin 2>/dev/null | wc -l")
    log "Nano shards present: $count / $EXPECTED_SHARDS"
    if [[ "$count" -ge "$EXPECTED_SHARDS" ]]; then
        log "All shards present — proceeding"
        break
    fi
    sleep 30
done

# Sync latest scripts and TSV to Nano
log "Syncing scripts and sweep TSV to Nano..."
rsync -az "$REPO_ROOT/scripts/run_experiment.sh" \
          "$REPO_ROOT/scripts/sweep_runner.sh" \
          "$REPO_ROOT/scripts/parse_log.py" \
          "$NANO_USER@$NANO_HOST:$NANO_ROOT/scripts/"
rsync -az "$REPO_ROOT/scripts/sweeps/nano_s01_arch.tsv" \
          "$NANO_USER@$NANO_HOST:$NANO_ROOT/scripts/sweeps/"
rsync -az "$REPO_ROOT/train_gpt_agx.py" \
          "$NANO_USER@$NANO_HOST:$NANO_ROOT/"

# Re-apply patches (idempotent)
ssh -i "$NANO_KEY" "$NANO_USER@$NANO_HOST" "
    sed -i 's|SCRIPT=\"\${SCRIPT:-train_gpt_sota_decoded.py}\"|SCRIPT=\"\${SCRIPT:-train_gpt_agx.py}\"|' $NANO_ROOT/scripts/run_experiment.sh
    sed -i 's|MAX_WALLCLOCK_SECONDS:-400|MAX_WALLCLOCK_SECONDS:-1800|' $NANO_ROOT/scripts/run_experiment.sh
" 2>/dev/null || true

log "Launching arch sweep on Nano..."
ssh -i "$NANO_KEY" "$NANO_USER@$NANO_HOST" "
    cd $NANO_ROOT && \
    nohup env TIMEOUT_SECS=2700 \
              MAX_WALLCLOCK_SECONDS=1800 \
              MAX_VAL_TOKENS=2097152 \
              DATA_DIR=$DATA_DIR \
              REPO_ROOT=$NANO_ROOT \
        bash scripts/sweep_runner.sh scripts/sweeps/nano_s01_arch.tsv \
        > logs/sweep/orchestrator_nano_s01.log 2>&1 & echo \"Nano sweep PID=\$!\"
"

log "=== nano_queue_s01 handed off to Nano — monitor via:"
log "    ssh ghostmini 'tail -f $NANO_ROOT/logs/sweep/orchestrator_nano_s01.log'"
