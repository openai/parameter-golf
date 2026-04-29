#!/usr/bin/env bash
# Deprecated safety shim.
#
# The PE ablation workflow must use scripts/run_pe_ablation.py so the RunPod
# bundle remains allowlist-only and does not upload helper scripts.  This shell
# wrapper is intentionally disabled to prevent accidental launches through the
# stale helper-script path.
set -euo pipefail

cat >&2 <<'EOF'
scripts/run_pe_ablation.sh is deprecated and intentionally disabled.
Use:
  python3 scripts/run_pe_ablation.py --dry-run

After reviewing the dry run and getting budget approval, launch via the Python
runner only. It generates the allowlisted on-pod train_gpt.py from the tracked
record source and bundles only allowlisted data/setup file names.
EOF
exit 2
