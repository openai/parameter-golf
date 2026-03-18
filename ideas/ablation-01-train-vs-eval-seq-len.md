# Ablation 1. Train Sequence Length Versus Eval Sequence Length

## Goal

Test whether this baseline is paying too much training cost for 1024-token attention.

## Why This Is One of the First Ablations

This directly tests my highest-conviction claim: the baseline appears optimization-limited under a fixed 600-second cap, and shorter training sequences may buy enough extra updates to improve final exact roundtrip `val_bpb`.

## Suggested Variants

- `TRAIN_SEQ_LEN=256 EVAL_SEQ_LEN=1024`
- `TRAIN_SEQ_LEN=512 EVAL_SEQ_LEN=1024`
- `TRAIN_SEQ_LEN=1024 EVAL_SEQ_LEN=1024`

Keep architecture and byte budget fixed.

## Decision Rule

If `256` or `512` training length wins on final exact roundtrip `val_bpb`, then train/eval sequence decoupling becomes the first baseline change to keep.

## Recommendation

Run first
