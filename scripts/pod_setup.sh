#!/usr/bin/env bash
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

exec bash "${REPO_ROOT}/scripts/install_cu124_fa3_env.sh"
