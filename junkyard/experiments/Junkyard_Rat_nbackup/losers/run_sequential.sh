#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env LOADER_MODE=sequential bash "${SCRIPT_DIR}/../run.sh"
