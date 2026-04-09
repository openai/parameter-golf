#!/bin/bash
# Run all 3 seeds for statistical significance
# Total time: ~30 min on 8xH100

set -e

echo "=== Running all 3 seeds for SOTA verification ==="
echo ""

# Seed 314 (best)
echo ">>> Seed 314"
export SEED=314
bash records/track_10min_16mb/2026-04-09_SP1024_TTT_ParallelRes_QK5/run_seed314.sh

# Seed 42
echo ""
echo ">>> Seed 42"
export SEED=42
bash records/track_10min_16mb/2026-04-09_SP1024_TTT_ParallelRes_QK5/run_seed314.sh

# Seed 999
echo ""
echo ">>> Seed 999"
export SEED=999
bash records/track_10min_16mb/2026-04-09_SP1024_TTT_ParallelRes_QK5/run_seed314.sh

echo ""
echo "=== All seeds complete ==="
echo "Check logs/run007_s*.log for results"
