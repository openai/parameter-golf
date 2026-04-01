# Title

Non-record: add HelixRecur v2 shared-depth recurrence submission

# Short Summary

This PR adds a single non-record submission folder for `records/track_non_record_16mb/2026-03-26_HelixRecur_v2` on top of `upstream/main`.

HelixRecur v2 keeps the HelixRecur recurrence stack compact with `6` shared blocks over `11` virtual passes and adds only `44` conditioning parameters to recover a small amount of depth-specific behavior. It is the active recurrence champion from the branch history, but it is being submitted honestly as exploratory non-record work rather than as a leaderboard attempt.

# Key Metrics

- Quick comparison used for judgment: `val_loss 7.54165596`, `val_bpb 4.46659346`, compressed `3042658`, total `3113435`, `step_avg 675.97ms`
- Longer non-record pass: `val_loss 4.63764717`, `val_bpb 2.74667588`, compressed `4224324`, total `4295101`, `step_avg 676.24ms`
- Code bytes: `70777`
- Parameter count: `15187040`
- Added trainable parameters vs v1: `44`

# Non-record Rationale

- This branch line remained far from the accepted SOTA, so it does not justify a record claim.
- The work is best framed as an exploratory recurrence result with a favorable quality-per-byte tradeoff inside its own local line.
- Later branch work did not produce a replacement winner: v3 lost on the longer pass, and the later v2 tournament did not dethrone v2.

# Why It Is Still Promising

- v2 materially improved HelixRecur v1 on the fair quick comparison while slightly shrinking total bytes.
- The model keeps a strong byte-efficiency story by reusing shared transformer blocks and spending only a negligible parameter budget on virtual-depth conditioning.
- The line established a cleaner recurrence baseline that survived both the v3 follow-up and the later micro-variant tournament.

# Next Direction

The next direction is gene-coded low-rank specialization: preserve the compact shared-depth recurrent base and add tiny learned low-rank specialization paths keyed by virtual depth, instead of stacking more scalar-only rescue knobs.
