# Non-record: Three Approaches — Lessons Learned

**Best legal result: 1.1188 BPB** (Approach B, s_0 TTT score only)

## Context

Previous PR #991 was closed because TTT re-scored tokens after training on them. This submission reports only the legal s_0 score (cumulative first-pass BPB where each token is scored before being used for training). All GPTQ calibration runs within the 600s training budget.

## Results

| Approach | Base | TTT? | val_bpb | Artifact | Status |
|----------|------|------|---------|----------|--------|
| **A** | #569 (VRL+LeakyReLU²+GPTQ) int5 | No | 1.1317 | <16MB | int5 penalty too high on d=512 |
| **B base** | #576 (d=576, 33.6M) int5 | No | 1.1249 | 15.3MB | Strong base, no TTT |
| **B + TTT** | #576 (d=576, 33.6M) int5 | s_0 only | **1.1188** | 15.3MB | Legal score-first, no re-eval |
| **C** | #505 (GEPA) int5 | s_0 only | N/A | 16.3MB | Artifact over limit |

## Key Lessons

1. **TTT re-scoring is illegal**: score→train→re-score reports s_1 which benefits from training on eval tokens. Only s_0 (cumulative first-pass) is legal.
2. **int5 penalty on d=512**: Switching #569 from int6 to int5 costs +0.014 BPB — the architecture was optimized for int6 precision.
3. **Legal s_0 TTT gives ~0.006 BPB**: B's base 1.1249 → s_0 1.1188 = -0.0061 improvement from backward-looking TTT.
4. **GEPA doesn't fit at int5**: 33.6M params at int5+3% prune+LZMA = 16.3MB. Would need 6%+ pruning or smaller model.
5. **GPTQ calibration timing matters**: Must complete within 600s training budget. Our script reserves 10-45s from training for calibration.

## Rule Compliance

- All GPTQ calibration within training budget (assert in code)
- All artifacts asserted < 16MB
- All eval times asserted < 600s
- TTT reports s_0 only — no second eval pass
- No val tokens in artifact

Based on PRs #569 (@gowtham0992), #576 (@cmcdnd), #505 (@JoeProAI).
