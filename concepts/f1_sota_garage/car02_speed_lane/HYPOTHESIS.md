# Car02: Speed Lane Hypothesis

## Question
Can we improve leaderboard BPB while staying legal and within runtime/size limits
by targeting architecture knobs that increase learning per wallclock step?

## Prediction
`t2_rope24` remains the strongest safe baseline in this lane, while long n-gram
eval passes are best treated as teacher diagnostics unless time-bounded.

## Isolation Rule
One variable per test. Every run must include an explicit hypothesis.

## Current Best Safe Signal
- `legal_ttt_exact val_bpb: 1.11906584` (`t2_rope24`, seed 1337)
- Artifact under 16MB
- Runtime profile compatible with record-track constraints

## Status
ACTIVE — primary race lane for speed-safe improvements.

## Verdict
_In progress._
