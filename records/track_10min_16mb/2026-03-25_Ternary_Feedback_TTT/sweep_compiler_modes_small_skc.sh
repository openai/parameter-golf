#!/usr/bin/env bash
# ============================================================================
# Compiler Mode Sweep for Small SKC Proxy Run
# Compares 'none', 'default', and 'reduce-overhead' throughput.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ORCHESTRATOR="${DIR}/orchestrate_small_skc_multigpu_runpod.sh"

[[ -f "$ORCHESTRATOR" ]] || { echo "ERROR: Orchestrator not found: $ORCHESTRATOR" >&2; exit 1; }

# Values to test
COMPILE_MODES=("none" "default" "reduce-overhead")

echo "=========================================================================="
echo "  Compiler Mode Sweep: $(date)"
echo "  Target Modes: ${COMPILE_MODES[*]}"
echo "=========================================================================="

# Use a fixed batch size for fair comparison (the new 32K default)
export TRAIN_BATCH_TOKENS=32768

for MODE in "${COMPILE_MODES[@]}"; do
    echo "--------------------------------------------------------------------------"
    echo "  RUNNING COMPILE_MODE=${MODE}"
    echo "--------------------------------------------------------------------------"
    
    export COMPILE_MODE=$MODE
    export FAST_SMOKE=1
    export MAX_WALLCLOCK_SECONDS=180  # Slightly longer to allow for compile time
    
    bash "$ORCHESTRATOR"
done

echo "=== SWEEP COMPLETE ==="
