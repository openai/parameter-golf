#!/bin/bash
set -euo pipefail

# Resume BW17 from the first failed rapid arm onward.
export RAPID_SOURCE_ARMS_CSV="${RAPID_SOURCE_ARMS_CSV:-BW17DGX-02,BW17DGX-03,BW17DGX-04,BW17DGX-05,BW17DGX-06,BW17DGX-07}"
export RAPID_CONTROL_INT6="${RAPID_CONTROL_INT6:-3.39016331}"

bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_ablation_sequence.sh"
