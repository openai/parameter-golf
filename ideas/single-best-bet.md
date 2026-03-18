# Single Best Bet

If I only tried one thing first, I would decouple train and evaluation sequence length and train at `512` or even `256` tokens while evaluating at `1024`.

## Why This Is My Best Bet

This is the cleanest way to convert wasted quadratic attention FLOPs into more optimizer progress inside the same 10-minute cap.

The main evidence is:

- the 10-minute baseline is still improving when it stops
- the same architecture continues improving far beyond that in the 4-hour run
- evaluation is already very cheap relative to the cap

That combination strongly suggests the model is not yet getting enough useful optimization during training. Shorter training sequence length directly attacks that bottleneck without increasing artifact size and without requiring a risky redesign.

## First Concrete Experiment

Implement `EVAL_SEQ_LEN`, then compare at fixed 600 seconds:

- `TRAIN_SEQ_LEN=256 EVAL_SEQ_LEN=1024`
- `TRAIN_SEQ_LEN=512 EVAL_SEQ_LEN=1024`
- current baseline `TRAIN_SEQ_LEN=1024 EVAL_SEQ_LEN=1024`

Pick the winner by final exact roundtrip `val_bpb`.
