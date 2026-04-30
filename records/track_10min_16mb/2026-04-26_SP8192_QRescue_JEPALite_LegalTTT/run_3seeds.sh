#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

./run_seed.sh 42
./run_seed.sh 314
./run_seed.sh 999
python update_submission_json.py train_seed42.log train_seed314.log train_seed999.log
