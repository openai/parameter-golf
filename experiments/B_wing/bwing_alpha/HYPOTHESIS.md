# B-WING ALPHA — Fix the Alpha Curve

## Hypothesis
Our alpha clamp (0.75) is leaving massive BPB on the table. PR #809 clips at 0.95,
meaning high-order n-gram matches can almost fully override the model. Combined with
a lower floor (0.05 vs our 0.20), confident model predictions stay clean while
uncertain tokens get aggressively n-gram'd.

## Changes from X-WING baseline
1. NGRAM_EVAL_ALPHA_MIN: 0.20 → 0.05
2. NGRAM_EVAL_ALPHA_MAX: 0.75 → 0.60
3. Alpha CLIP max: 0.75 → 0.95 (in the cubric clip line)
4. Keep cubric 3D adaptive system and warm starts

## Expected impact
The alpha clip alone should be worth 0.05-0.10 BPB.
The floor fix prevents over-mixing on confident model tokens.

## What NOT to change
- Keep our cubric 3D system (they don't have it, this is our edge)
- Keep our architecture, training, everything else identical
- Keep entropy center at 3.0 (same as theirs)
