#!/usr/bin/env bash
# Orchestrator: runs stages 0-3 automatically, then stops for human decision.
# Stage 4 (04a or 04b) is never auto-triggered — inspect results first.
#
# Usage:
#   cd /workspace/parameter-golf
#   bash scripts/runpod_pipeline/run_all.sh
set -euo pipefail

if [ -z "${EXPECTED_SHA:-}" ]; then
    echo "WARNING: EXPECTED_SHA unset — 00_verify_pod.sh will run ancestry-only (no exact-SHA pin)." >&2
    echo "  For Session launches, export EXPECTED_SHA=<launch-sha> before invoking run_all.sh." >&2
fi

REPO_DIR="/workspace/parameter-golf"
PIPELINE_DIR="${REPO_DIR}/scripts/runpod_pipeline"

die() { echo ""; echo "PIPELINE ABORT: $*" >&2; exit 1; }

announce() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  ▶  $*"
    echo "  Started: $(date +%H:%M:%S)"
    echo "══════════════════════════════════════════════════════════════"
}

finish() {
    echo "  ✓  $*  [$(date +%H:%M:%S)]"
}

cd "${REPO_DIR}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RunPod Pipeline — PR #1610 + Posterior Corrector           ║"
echo "║  Branch: submission/pr1610-corrector                        ║"
echo "║  SHA:    ${EXPECTED_SHA:-a33191f572430566b88c4d61badb0369e1e6f9a3}           ║"
echo "║  Budget: ~\$40 for stages 0-4 (~6 hrs at \$21.52/hr)         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Stages 0-3 will run automatically."
echo "Pipeline will STOP before Stage 4 for your review."
echo ""

announce "Stage 0: Pod verification"
bash "${PIPELINE_DIR}/00_verify_pod.sh" || die "Stage 0 failed. Log: runs/00_verify_pod.log"
finish "Stage 0: OK"

announce "Stage 1: Dataset download"
bash "${PIPELINE_DIR}/01_download_data.sh" || die "Stage 1 failed. Log: runs/01_download_data.log"
finish "Stage 1: OK"

announce "Stage 2: Gate A (seed 0 train+eval)"
bash "${PIPELINE_DIR}/02_gate_a.sh" || die "Stage 2 failed. Log: runs/seed0_log.txt"
finish "Stage 2: OK"

announce "Stage 3: Corrector ablations (1a / 1b / 1c)"
bash "${PIPELINE_DIR}/03_ablations.sh" || die "Stage 3 failed. Log: runs/03_ablations.log"
finish "Stage 3: OK"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AUTO-PIPELINE COMPLETE (Stages 0–3)                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results summary:"
echo "  Gate A:    $(python3 -c "import json; d=json.load(open('runs/gate_a_summary.json')); print(f\"seed0 BPB={d['bpb']:.8f}  status={d['status']}\")" 2>/dev/null || echo "(see runs/gate_a_summary.json)")"
echo ""
echo "Ablation deltas:"
/opt/pg-venv/bin/python3 - runs/ablation_summary.json 2>/dev/null <<'PY' || echo "  (see runs/ablation_summary.json)"
import json, sys
s = json.load(open(sys.argv[1]))
for lbl, r in sorted(s["runs"].items()):
    marker = " ← best" if lbl == s["best_config"] else ""
    print(f"  {lbl}: alpha={r['config']['CORRECTOR_ALPHA']} orders={r['config']['CORRECTOR_ORDERS']:<8} "
          f"delta={r['delta']:+.6f}{marker}")
print(f"\n  Recommended: {s['recommended_path'].upper()}")
PY
echo ""
echo "Next steps:"
echo ""
echo "  1. Review recommendation:"
echo "       bash scripts/runpod_pipeline/04_decide_and_proceed.sh"
echo ""
echo "  2a. If PRIMARY path:"
echo "       BEST_ALPHA=<X> BEST_ORDERS='<Y>' bash scripts/runpod_pipeline/04a_gate_b.sh"
echo ""
echo "  2b. If FALLBACK path:"
echo "       bash scripts/runpod_pipeline/04b_fallback_level1a.sh"
echo ""
echo "  3. Preserve artifacts (always run before pod termination):"
echo "       UPLOAD_TARGET=hf:<repo>:<path> bash scripts/runpod_pipeline/05_preserve_artifacts.sh"
echo ""
