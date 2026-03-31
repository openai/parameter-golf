#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env MLP_SCALE_INIT=1.02 bash "${SCRIPT_DIR}/../run_pop_test.sh"
