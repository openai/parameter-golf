# 1. Train Shorter Than You Evaluate

## Core Thesis

`TRAIN_SEQ_LEN=1024` is probably the wrong place to spend FLOPs for a model that is still far from converged under a 600-second cap.

## What Bottleneck It Attacks

This attacks update count under a fixed wallclock budget. The current script ties training and validation to the same sequence length, and attention cost grows quadratically with sequence length.

Relevant code:

- `TRAIN_SEQ_LEN` default: `train_gpt.py:58`
- Validation currently uses `args.train_seq_len`: `train_gpt.py:235-257`
- Train loader also uses `args.train_seq_len`: `train_gpt.py:946`, `train_gpt.py:1014`

## Why It Should Improve `val_bpb`

The shipped baseline is still improving when it stops at the 10-minute wallclock cap:

- baseline stop: `step:13780/20000 val_bpb:1.2172`

The 4-hour run with the same 9x512 layout keeps improving much further:

- 4-hour stop: `step:329430/500000 val_bpb:1.1749` before export roundtrip

That pattern says this baseline is still optimization-limited. If so, extra optimizer updates are likely worth more than training with 1024-token attention on every step. Since evaluation only takes around 1.4 seconds, there is no reason to force training and evaluation to use the same sequence length.

## Expected Effect

- Training speed: much faster if training seq length drops to `512` or `256`
- Evaluation speed: unchanged if eval stays at `1024`, or slightly slower if you make eval more expensive
- Compressed artifact size: unchanged

## Difficulty

2/5

## Rule-Risk

1/5

## Smallest Decisive Experiment

Add `EVAL_SEQ_LEN` and run fixed-600-second comparisons:

- `TRAIN_SEQ_LEN=256 EVAL_SEQ_LEN=1024`
- `TRAIN_SEQ_LEN=512 EVAL_SEQ_LEN=1024`
- `TRAIN_SEQ_LEN=1024 EVAL_SEQ_LEN=1024`

Hold architecture fixed and compare final exact roundtrip `val_bpb`.

## Recommendation Bucket

Baseline script improvement
