# B-WING FULL PORT — All #809 N-gram Techniques

## Hypothesis
Combine all three key innovations from PR #809 onto our X-WING base:
1. Alpha curve: min=0.05, max=0.60, clip=0.95
2. Per-order entropy center shift: -0.25*(order - min_order)
3. Fixed order multipliers: (0.3, 0.3, 0.97, 2.0, 2.0, 2.0, 2.0, 2.0)
   → replaces cubric 3D adaptive system

This is the "kitchen sink" variant. If bwing_alpha and bwing_entropy_shift
each show gains, this should stack them.

## Changes from X-WING baseline
1. NGRAM_EVAL_ALPHA_MIN: 0.20 → 0.05
2. NGRAM_EVAL_ALPHA_MAX: 0.75 → 0.60
3. Alpha CLIP: 0.75 → 0.95
4. Per-order entropy center shift
5. Fixed order multipliers replacing cubric 3D
6. Order 4 mult: 0.45 → 0.97 (big change)
7. Order 2 mult: 0.45 → 0.30

## Risk
Removing cubric 3D loses per-entropy-bin adaptation. But their fixed mults
work at 0.295 BPB so the risk is bounded.

## Expected impact
Should approach their 0.295 while keeping our better base model (~1.12 vs 1.14).
Target: sub-0.30 BPB.
