# Frontier scan 2 — 2026-04-21

**Mode:** research
**Type:** incremental scan (scan 2 of the day)

## Summary

Quiet scan. One new PR (#1752) since the morning scan. No change to leaderboard.

## PR #1752 — luccifer00, "healing-phase training submission" (1.205 val_bpb)

Non-competitive non-record submission. Author applies a "healing phase" near the
end of training (reduced LR + WD) combined with factorized late layers. Claimed
1.205 val_bpb is far above SOTA and our baseline. No actionable signal.

The healing-phase concept overlaps in spirit with spec 011 (tapered WD) but
combined with LR reduction. Could be interesting at competitive scores; at 1.205
it's not evidence of anything.

## Leaderboard status

Unchanged from scan 1 (earlier today):

- **We are the legal leaderboard leader** at 1.06549 (PR #1736, our baseline)
- Nearest clean PR: #1676 @ 1.0788 (+0.013 behind us)
- Disputed frontier: #1738 @ 1.0354 (pre-quant TTT, likely-illegal)

## Meta

Three null/regression runs today (specs 011, 013, 014). The community hasn't
posted anything new that threatens our position. The open question remains
whether we can push our own baseline — not whether someone is beating us.
