#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${DIR}/runpod_queue_stage0_core.sh"
"${DIR}/runpod_queue_stage1_core.sh"
"${DIR}/runpod_queue_stage2_core.sh"
"${DIR}/runpod_queue_stage3_sweeps.sh"
"${DIR}/runpod_queue_stage4_seeds.sh"
