#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env RESID_MIX_X_INIT=0.98 RESID_MIX_X0_INIT=0.02 bash "${SCRIPT_DIR}/../run_pop_test.sh"
