# Local Baseline Reproduction

## Summary

This submission records a local single-GPU reproduction of the OpenAI Parameter Golf NaiveBaseline.

## Environment

- local single-GPU run
- train_shards=1
- seq_len=1024
- grad_accum_steps=8

## Result

- best observed val_bpb: **1.3529 @ step 4200**

## Notes

- validation improved from 4.1077 to 1.3529
- performance plateaued after around step 4200
- this is a local baseline reproduction and not a leaderboard SOTA submission
