#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Run all cadence ablation arms — sequential, 8 GPUs per arm
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash experiments/run_all.sh          # all 8 arms (~20 min)
#   bash experiments/run_all.sh H1       # H1 only: 4f2cx2 cadence 1-4
#   bash experiments/run_all.sh H2       # H2 only: 3f3cx2 cadence 1-4
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

FRONT="${1:-ALL}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="experiments/ablation_run_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════════════════"
echo "CADENCE ABLATION — Front: $FRONT — $(date)"
echo "Log: $LOGFILE"
echo "═══════════════════════════════════════════════════════════════"

PASS=0
FAIL=0
TOTAL=0

run_arm() {
    local script="$1"
    local arm_name=$(basename "$script" .sh)
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "[$TOTAL] $arm_name — started $(date +%H:%M:%S)"
    echo "────────────────────────────────────────────────────────"

    if bash "$script" 2>&1 | tee -a "$LOGFILE"; then
        echo "[$arm_name] PASSED"
        PASS=$((PASS + 1))
    else
        echo "[$arm_name] FAILED (exit $?)"
        FAIL=$((FAIL + 1))
    fi
}

# ── H1: Cadence characterization on 4f+2cx2 ──
if [ "$FRONT" = "ALL" ] || [ "$FRONT" = "H1" ]; then
    echo ""
    echo "══ H1: Cadence sweep on 4f+2cx2 (RC-0) ══"
    run_arm experiments/H1_cadence_characterization/4f2cx2_cad1_025.sh
    run_arm experiments/H1_cadence_characterization/4f2cx2_cad2_025.sh
    run_arm experiments/H1_cadence_characterization/4f2cx2_cad3_025.sh
    run_arm experiments/H1_cadence_characterization/4f2cx2_cad4_025.sh
fi

# ── H2: Cadence × architecture on 3f+3cx2 ──
if [ "$FRONT" = "ALL" ] || [ "$FRONT" = "H2" ]; then
    echo ""
    echo "══ H2: Cadence sweep on 3f+3cx2 (6x2) ══"
    run_arm experiments/H2_cadence_x_architecture/3f3cx2_cad1_025.sh
    run_arm experiments/H2_cadence_x_architecture/3f3cx2_cad2_025.sh
    run_arm experiments/H2_cadence_x_architecture/3f3cx2_cad3_025.sh
    run_arm experiments/H2_cadence_x_architecture/3f3cx2_cad4_025.sh
fi

# ── Summary ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "ABLATION COMPLETE — $PASS passed, $FAIL failed, $TOTAL total"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Results:"
echo "  H1: experiments/H1_cadence_characterization/results/"
echo "  H2: experiments/H2_cadence_x_architecture/results/"
echo ""
echo "Full log: $LOGFILE"

if [ "$FAIL" -gt 0 ]; then
    echo "WARNING: $FAIL arm(s) failed. Check log for details."
    exit 1
fi
