#!/bin/bash
# Documentation-only helper mirroring bootstrap's seed order.
set -euo pipefail
for SEED in 42 1337 2024; do
  export SEED
  bash run_seed.sh
done
