#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${DIR}/runpod_baseline.sh"
"${DIR}/runpod_gate.sh"
"${DIR}/runpod_flop_push.sh"
