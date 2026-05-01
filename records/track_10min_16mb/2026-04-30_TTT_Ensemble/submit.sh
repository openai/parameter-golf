#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE="ubuntu@91.239.86.233"

echo "Waiting for seed 314 to finish..."
while true; do
  LOGFILE=$(ssh $REMOTE 'for f in ~/parameter-golf/logs/*.txt; do
    seed=$(grep "^seed: 314" "$f" 2>/dev/null | head -1)
    if [ -n "$seed" ]; then echo "$f"; break; fi
  done')
  if [ -z "$LOGFILE" ]; then
    echo "  No seed 314 log found yet, waiting 30s..."
    sleep 30
    continue
  fi
  DONE=$(ssh $REMOTE "grep -c '^quantized_ttt_phased' $LOGFILE 2>/dev/null || echo 0")
  if [ "$DONE" -ge 1 ]; then
    echo "  Found completed log: $LOGFILE"
    break
  fi
  echo "  Log exists but not finished, waiting 30s..."
  sleep 30
done

echo "Pulling log..."
scp "$REMOTE:$LOGFILE" "$DIR/seed314.log"

echo "Parsing results..."
ENS_BPB=$(grep '^peer_ens:ensemble_covered_only' "$DIR/seed314.log" | grep -oE 'val_bpb:[0-9.]+' | cut -d: -f2)
BASELINE_BPB=$(grep '^peer_ens:baseline' "$DIR/seed314.log" | grep -oE 'val_bpb:[0-9.]+' | cut -d: -f2)
EVAL_TIME=$(grep '^quantized_ttt_phased' "$DIR/seed314.log" | grep -oE 'eval_time:[0-9.]+ms' | cut -d: -f2 | sed 's/ms//')
PRE_QUANT=$(grep '^diagnostic pre-quantization' "$DIR/seed314.log" | grep -oE 'val_bpb:[0-9.]+' | cut -d: -f2)
POST_QUANT=$(grep '^diagnostic quantized' "$DIR/seed314.log" | grep -oE 'val_bpb:[0-9.]+' | cut -d: -f2)
ARTIFACT=$(grep '^Total submission size' "$DIR/seed314.log" | grep -oE '[0-9]+ bytes' | cut -d' ' -f1)

PRE_Q5=$(printf "%.5f" "$PRE_QUANT")
POST_Q5=$(printf "%.5f" "$POST_QUANT")
ENS_5=$(printf "%.5f" "$ENS_BPB")
DELTA=$(python3 -c "print(f'{$ENS_BPB - 1.05855:.5f}')")

echo "Results:"
echo "  Pre-quant BPB:  $PRE_Q5"
echo "  Post-quant BPB: $POST_Q5"
echo "  Ensemble BPB:   $ENS_5"
echo "  Baseline BPB:   $BASELINE_BPB"
echo "  Eval time:      ${EVAL_TIME}ms"
echo "  Artifact bytes: $ARTIFACT"
echo "  Delta vs #2014: $DELTA"

echo "Updating README.md..."
sed -i '' "s/| 314  | -   | -  | - | - |/| 314  | $PRE_Q5 | $POST_Q5 | **$ENS_5** | $ARTIFACT |/" "$DIR/README.md"
sed -i '' "s/\*\*val_bpb = TBD\*\* (1 seed)/**val_bpb = $ENS_5** (1 seed)/" "$DIR/README.md"
sed -i '' "s/Delta: TBD/Delta: $DELTA vs PR #2014 baseline (1.05855)/" "$DIR/README.md"

echo "Updating submission.json..."
python3 -c "
import json
with open('$DIR/submission.json') as f:
    d = json.load(f)
d['val_bpb'] = round($ENS_BPB, 5)
d['val_bpb_std'] = None
d['seed_results'] = {'314': {'val_bpb': round($ENS_BPB, 5), 'artifact_bytes': $ARTIFACT}}
with open('$DIR/submission.json', 'w') as f:
    json.dump(d, f, indent=2)
    f.write('\n')
"

echo "Committing and pushing..."
cd "$DIR/../../.."
git add records/track_10min_16mb/2026-04-30_TTT_Ensemble/README.md \
        records/track_10min_16mb/2026-04-30_TTT_Ensemble/submission.json \
        records/track_10min_16mb/2026-04-30_TTT_Ensemble/seed314.log
git commit -m "seed 314 results: val_bpb=$ENS_5, K=3 peer-LoRA ensemble"
git push fork ttt-ensemble-clean

echo ""
echo "DONE! val_bpb=$ENS_5 eval_time=${EVAL_TIME}ms delta=$DELTA"
