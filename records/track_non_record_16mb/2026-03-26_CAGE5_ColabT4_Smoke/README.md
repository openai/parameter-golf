# CAGE5 Colab T4 smoke (non-record 16MB)

This folder captures a non-record smoke submission for Parameter Golf.

This is an in-progress submission used to validate a complete training -> quantization/export -> evaluation pipeline for a strictly causal hashed 5-gram mixer, and then verify that the same mixer also stacks with legal score-first TTT.

## Summary

- Hardware: 1x Tesla T4 (Google Colab GPU)
- Track: non-record-16mb
- Core idea: interpolate the neural model with a strictly causal hashed 5-gram cache during sliding-window evaluation and legal score-first TTT evaluation

## Best result in `train.log`

- `legal_ttt_exact val_loss: 4.43143776`
- `legal_ttt_exact val_bpb: 2.56285268`
- `final_int6_sliding_window_exact val_loss: 4.44510223`
- `final_int6_sliding_window_exact val_bpb: 2.57075530`
- `Total submission size int6+lzma: 1315287 bytes`
- `Serialized model int6+lzma: 1219864 bytes`
- `Code size: 95423 bytes`

## A/B result against baseline (`ablation_baseline.log`)

- Baseline with legal TTT, no n-gram: `legal_ttt_exact val_bpb = 2.92123914`
- Legal TTT + n-gram: `legal_ttt_exact val_bpb = 2.56285268`
- Absolute gain: `0.35838646 BPB`

## Included files

- `train_gpt.py` — Colab-tested script used for the winning smoke run
- `train.log` — winning legal TTT + n-gram run
- `ablation_baseline.log` — matched baseline without n-gram
- `submission.json` — metadata for this non-record smoke submission
- `README.md` — summary and results
