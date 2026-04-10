#!/usr/bin/env bash
# ============================================================================
# Batch Size Sweep for Small SKC Proxy Run
# Investigates relationship between batch size, step latency, and throughput.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ORCHESTRATOR="${DIR}/orchestrate_small_skc_multigpu_runpod.sh"

[[ -f "$ORCHESTRATOR" ]] || { echo "ERROR: Orchestrator not found: $ORCHESTRATOR" >&2; exit 1; }

# Values to test: explore the Pareto frontier of GPU saturation
BATCH_SIZES=(8192 16384 32768 65536)

echo "=========================================================================="
echo "  Simplified Batch Size Convergence Sweep: $(date)"
echo "  Target Sizes: ${BATCH_SIZES[*]}"
echo "  Compiler Mode: none (EAGER)"
echo "  Duration: 10 minutes per configuration"
echo "=========================================================================="

for BS in "${BATCH_SIZES[@]}"; do
    echo "--------------------------------------------------------------------------"
    echo "  RUNNING BATCH_SIZE=${BS}"
    echo "--------------------------------------------------------------------------"
    
    export TRAIN_BATCH_TOKENS=$BS
    export COMPILE_MODE="none"            # Eager mode, no warmup
    export FAST_SMOKE=0                   # Full training phase
    export MAX_WALLCLOCK_SECONDS=600      # 10 minutes for convergence
    
    if bash "$ORCHESTRATOR"; then
        echo "  [SUCCESS] BATCH_SIZE=${BS}"
    else
        echo "  [FAILED] BATCH_SIZE=${BS} - Continuing sweep..."
    fi
    
    # After each run, metrics are in the local artifacts directory created by the orchestrator.
    # We can extract the latest tokens/sec from the log.
done

echo "=== SWEEP COMPLETE ==="
echo "Please review the individual logs in the artifacts directories."
