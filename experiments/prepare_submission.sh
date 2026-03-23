#!/bin/bash
# Package an experiment into a proper submission folder
# Usage: ./prepare_submission.sh <experiment_dir> <submission_name> <author> <github_id>
# Example: ./prepare_submission.sh idea01_byte_weighted_loss "ByteWeightedLoss_11L_Int5MLP" "myname" "mygithub"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

EXPERIMENT="${1:?Usage: $0 <experiment_dir> <submission_name> <author> <github_id>}"
NAME="${2:?Need submission name}"
AUTHOR="${3:?Need author name}"
GITHUB_ID="${4:?Need github id}"

EXP_DIR="$SCRIPT_DIR/$EXPERIMENT"
DATE=$(date +%Y-%m-%d)
SUBMISSION_DIR="$REPO_DIR/records/track_10min_16mb/${DATE}_${NAME}"

if [ -d "$SUBMISSION_DIR" ]; then
    echo "ERROR: $SUBMISSION_DIR already exists"
    exit 1
fi

mkdir -p "$SUBMISSION_DIR"

# Copy train script and logs
cp "$EXP_DIR/train_gpt.py" "$SUBMISSION_DIR/"
for log in "$EXP_DIR"/train_seed*.log; do
    [ -f "$log" ] && cp "$log" "$SUBMISSION_DIR/"
done

# Extract scores from logs
SCORES=""
TOTAL_BPB=0
COUNT=0
for seed in 42 1337 2024; do
    LOG="$EXP_DIR/train_seed${seed}.log"
    if [ -f "$LOG" ]; then
        BPB=$(grep "final_int8_zlib_roundtrip_exact" "$LOG" | grep -oP 'val_bpb:\K[0-9.]+' || echo "")
        BYTES=$(grep "Serialized model int6" "$LOG" | grep -oP '[0-9]+(?= bytes)' | head -1 || echo "")
        if [ -n "$BPB" ]; then
            SCORES="$SCORES| $seed | $BPB | $BYTES | yes |\n"
            TOTAL_BPB=$(python3 -c "print($TOTAL_BPB + $BPB)")
            COUNT=$((COUNT + 1))
        fi
    fi
done

if [ "$COUNT" -gt 0 ]; then
    MEAN_BPB=$(python3 -c "print(f'{$TOTAL_BPB / $COUNT:.5f}')")
else
    MEAN_BPB="N/A"
fi

# Generate README
cat > "$SUBMISSION_DIR/README.md" << EOF
# $NAME

**val_bpb: $MEAN_BPB** (mean of $COUNT seeds, sliding window stride=64, post int5/int6+zstd quantization roundtrip)

## Run Command

\`\`\`bash
bash prepare.sh
SEED=42 bash eval/eval.sh
\`\`\`

## ${COUNT}-Seed Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|---------------|-------|
$(echo -e "$SCORES")| **Mean** | **$MEAN_BPB** | | |

## Key Techniques

(Fill in technique description here)

## Architecture

(Fill in architecture details here)
EOF

# Generate submission.json
cat > "$SUBMISSION_DIR/submission.json" << EOF
{
  "name": "$NAME",
  "val_loss": $MEAN_BPB,
  "bytes_total": 16000000,
  "blurb": "$NAME submission",
  "author": "$AUTHOR",
  "github_id": "$GITHUB_ID",
  "date": "$DATE"
}
EOF

echo "Submission prepared at: $SUBMISSION_DIR"
echo "  Mean BPB: $MEAN_BPB ($COUNT seeds)"
echo ""
echo "TODO: Edit README.md and submission.json with proper details"
