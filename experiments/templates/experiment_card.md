# EXP-XXX: [Title]

## Metadata
- **Date**: YYYY-MM-DD
- **Branch**: exp/[name]
- **Parent**: exp/reproduce-sota
- **Priority**: P0/P1/P2
- **Estimated runs**: X dev + X full + 3 seed
- **Estimated cost**: ~$X

## Hypothesis
[Why you expect this change to improve BPB. Be specific about the mechanism.]

## Null Hypothesis
[What would prove the change doesn't help or actively hurts.]

## Control Variables (what stays the same)
- [List everything that remains unchanged from the parent branch]

## Independent Variable (what changes)
- [The specific code/config change being tested]

## Success Criteria
- BPB < [current SOTA]
- Artifact size <= 15.95 MB
- No training slowdown > X ms/step

## Abort Criteria
- Artifact > 16 MB
- BPB > [threshold] on dev run
- Training > [threshold] ms/step

## Run Plan
1. DEV RUN: seed=314, 8xH100
   - Goal: verify compilation, artifact size, rough BPB
2. FULL RUN: seed=314, 8xH100, full 600s
   - Goal: real BPB, compare to reproduced baseline
3. DECISION GATE: if BPB < reproduced baseline, proceed to 3-seed
4. SEED RUNS: seeds 314, 42, 999
   - Goal: statistical significance (p < 0.01)

## Results

### Dev Run
- Date:
- BPB:
- Artifact size:
- Steps / ms per step:
- Notes:

### Full Run
- Date:
- BPB:
- Artifact size:
- Decision: PROCEED / ABORT / MODIFY

### 3-Seed Validation
| Seed | Steps | ms/step | BPB | Artifact |
|------|-------|---------|-----|----------|
| 314  |       |         |     |          |
| 42   |       |         |     |          |
| 999  |       |         |     |          |
| Mean |       |         |     |          |
| Std  |       |         |     |          |

### Statistical Test
- Welch's t-test: t=, df=, p=
- Significant: YES/NO

## Post-Mortem
### What happened:
### What surprised you:
### What to carry forward:
### What to drop:
