#!/usr/bin/env bash
set -euo pipefail

bash run_compile_bench.sh
bash run_init_ablation.sh
bash run_capacity_scout.sh
bash run_lambda_sweep.sh
