# Hypothesis

## Objective
Beat PR #779's 0.6683 BPB by adding cubric per-order adaptive alpha scaling to their BackoffNgramMixer.

## Single Change
- Add cubric: per-order multipliers on the entropy-adaptive alpha, boosting high-order (5-7) matches and suppressing low-order (2-3) noise. Proven on Podracer green (0.9357 vs 0.962 baseline = -0.026).

## Why It Might Work
- PR #779 uses flat alpha for all orders. But orders 5-7 consistently beat the model at higher rates than orders 2-3. Cubric differentiates the signal.
- Proven in Podracer green: multipliers converge to {2:0.3, 3:0.3, 4:1.0, 5:2.0, 6:2.0, 7:2.0}.
- Conservative estimate: 2.7% relative improvement on their 0.6712 mixer-only → ~0.654.

## Risks
- Green2 showed wider caps (4.0) catastrophically hurt. Must stay at ceiling=2.0.
- Alpha clip must stay ≤0.70. Effective max = 0.70 × 2.0 = 1.40 (proven safe).
- Cubric c-steps fire per-rank independently (not synchronized). Should converge similarly.

## Success Criteria
- Beat 0.6683 mean BPB (PR #779's 3-seed mean).

## Run Plan
- Seed 1337 first (matches their ablation baseline).
- 2 additional seeds for variance.
