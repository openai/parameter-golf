#!/usr/bin/env bash
# submit.sh — Package the best v10 script for H100 submission.
#
# Usage:
#   bash submit.sh [safe|moonshot]   # default: safe
#
# What it does:
#   1. Copies chosen train_gpt_v10_*.py → train_gpt.py in a staging dir
#   2. Runs on 1xH100 to check artifact size and quant_penalty
#   3. Prints instructions for 8xH100 final run
#
# IMPORTANT: This script DOES NOT auto-submit. Review output before submitting.

set -euo pipefail

VARIANT="${1:-safe}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

if [[ "$VARIANT" == "moonshot" ]]; then
    SRC="$SCRIPT_DIR/train_gpt_v10_moonshot.py"
    SUBMISSION_DIR="$REPO_ROOT/records/track_10min_16mb/2026-03-26_sunnypatneedi_moonshot_final"
    EXPECTED_BYTES=14000000   # 14MB target
else
    SRC="$SCRIPT_DIR/train_gpt_v10_safe.py"
    SUBMISSION_DIR="$REPO_ROOT/records/track_10min_16mb/2026-03-26_sunnypatneedi_safe_final"
    EXPECTED_BYTES=15500000   # 15.5MB target
fi

echo "====================================================="
echo " v10 Submit Script — variant: $VARIANT"
echo "====================================================="
echo " Source:       $SRC"
echo " Staging dir:  $SUBMISSION_DIR"
echo " Size target:  <$EXPECTED_BYTES bytes"
echo "====================================================="
echo ""

# ── Validate source exists ────────────────────────────────────────────────────
if [[ ! -f "$SRC" ]]; then
    echo "ERROR: Source script not found: $SRC"
    exit 1
fi

# ── Create staging dir ────────────────────────────────────────────────────────
mkdir -p "$SUBMISSION_DIR"
cp "$SRC" "$SUBMISSION_DIR/train_gpt.py"
echo "Copied $SRC → $SUBMISSION_DIR/train_gpt.py"

# ── Check line count ──────────────────────────────────────────────────────────
LINES=$(wc -l < "$SUBMISSION_DIR/train_gpt.py")
echo "Script lines: $LINES"

# ── Check script compresses well (pre-run estimate) ───────────────────────────
SCRIPT_BYTES=$(wc -c < "$SUBMISSION_DIR/train_gpt.py")
echo "Script size (raw): $SCRIPT_BYTES bytes"
SCRIPT_ZSTD_BYTES=$(zstd -19 --compress "$SUBMISSION_DIR/train_gpt.py" -c | wc -c 2>/dev/null || echo "unknown")
echo "Script size (zstd-19): $SCRIPT_ZSTD_BYTES bytes (estimate)"

echo ""
echo "====================================================="
echo " NEXT STEPS"
echo "====================================================="
echo ""
echo "1. On 1xH100 (quick smoke — ~2 min):"
echo "   cd $SUBMISSION_DIR"
echo "   nohup bash -c 'torchrun --standalone --nproc_per_node=1 train_gpt.py > smoke.log 2>&1' &"
echo "   tail -f smoke.log"
echo ""
echo "   After run — check artifact size:"
echo "   ls -lh artifact*.ptz 2>/dev/null || ls -lh final_model*.ptz 2>/dev/null"
echo "   python3 -c \"import os; s=os.path.getsize('$(ls $SUBMISSION_DIR/*.ptz 2>/dev/null | head -1 || echo ARTIFACT.ptz)')+os.path.getsize('$SUBMISSION_DIR/train_gpt.py'); print(f'Artifact: {s:,} bytes ({100*s/16000000:.1f}% of limit)')\""
echo ""
echo "2. If size < 16MB and quant_penalty < 0.005:"
echo "   → Run on 8xH100 (final validation, ~10 min):"
echo "   nohup bash -c 'torchrun --standalone --nproc_per_node=8 train_gpt.py > final.log 2>&1' &"
echo "   tail -f final.log"
echo ""
echo "3. Record val_bpb from final.log, then create submission.json:"
cat << 'JSONTEMPLATE'
   {
     "val_bpb": <FILL_IN>,
     "artifact_bytes": <FILL_IN>,
     "seeds": [42, 43, 44],
     "track": "10min_16mb",
     "description": "v10 $VARIANT: 11-layer GPT + XSA + Hedge Mixer + 11-gram + adaptive quant"
   }
JSONTEMPLATE
echo ""
echo "4. Submit PR following records/track_10min_16mb/SUBMISSION_GUIDE.md"
echo ""
echo "REMINDER: Always verify val_bpb with 3 seeds before claiming SOTA!"
echo "====================================================="
