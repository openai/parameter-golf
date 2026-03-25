# H8: Weight Sharing Isolation — Is the Crawler a Useful Regularizer?

## Question
Does weight-shared depth (crawler looping 2x) improve BPB over equivalent
unique layers, independent of recursion (C-steps)?

## Prediction
The cad0 result (1.1325) uses crawler blocks that loop twice but never
double-fire. The weight sharing forces the model to learn transformations
that work when applied twice — a form of regularization. On a
capacity-starved small model, this regularization should help by preventing
overfitting. On a large model, it may just limit capacity.

At 8L/384d (small, fast), weight sharing should help.
At 11L/512d (GS v7 scale), it may be neutral.

## Arms (0.25 scale)
| Arm | Config | Effective depth | Unique params |
|-----|--------|----------------|---------------|
| A | 8 unique flat layers | 8 | 8 blocks |
| B | 6 unique + 1 shared × 2 loops | 8 | 7 blocks |

Same effective depth. B has fewer unique parameters but weight-shared
extra depth. Both at cadence 0 (no C-steps).

## Implementation
Arm A: NUM_LAYERS=8, no crawler bank
Arm B: NUM_LAYERS=6, crawler bank enabled, loops=2
Need to adjust the GS v7 script to support fewer base layers + bank.

## Diagnostic Focus
- sliding_window BPB: does weight sharing help at small scale?
- Artifact size: B should be smaller (fewer unique weights)
- Per-step learning at val@500: is weight sharing helping quality per step?

## Status
NEEDS CODE CHANGE — must support NUM_LAYERS=6 + crawler bank = 8 effective.

## Verdict
_To be filled after runs._
