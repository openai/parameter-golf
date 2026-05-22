#!/usr/bin/env bash
# squid_monitor_watch.sh — continuous squid monitor loop for VS Code task panel.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL_SECS="${SQUID_MONITOR_INTERVAL_SECS:-15}"

while true; do
  "$ROOT_DIR/scripts/squid_monitor.sh"
  echo ""
  sleep "$INTERVAL_SECS"
done
