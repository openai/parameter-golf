#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${DIR}/runpod_spine_a.sh"
"${DIR}/runpod_spine_b.sh"
