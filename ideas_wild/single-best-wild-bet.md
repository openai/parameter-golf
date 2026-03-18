# Single Best Wild Bet

If I had to pick one moonshot, I would try a recurrent transformer with only `2` to `4` unique blocks, unrolled many times, with per-step gates or scales and one extra evaluation-time refinement pass.

## Why This Is My Best Wild Bet

It attacks the challenge at the right level:

- converts bytes into compute
- uses the artifact cap more efficiently than fully untied depth
- uses the currently underexploited evaluation-time budget
- pairs naturally with lightweight per-step modulation

This is the cleanest high-variance idea that still has a coherent path to beating the baseline for the right reasons.

## First Concrete Experiment

Build a version with:

- `3` unique blocks
- `9` unrolled applications
- per-step scale vectors
- one extra evaluation-only refinement pass

Then widen until the artifact returns near the current size budget and compare exact roundtrip `val_bpb` under the same 600-second training cap.
