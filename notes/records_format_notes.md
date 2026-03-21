# Records Format Notes

## Typical records layout

Observed structure in this repository:

- `records/track_10min_16mb/<date_name>/`
- `records/track_non_record_16mb/<date_name>/`

Typical files inside one run directory:

- `README.md`
- `submission.json`
- `train_gpt.py`
- one or more training logs such as `train.log`, `train_seed1338.log`, `train_v0.txt`
- sometimes extra artifacts such as `ablations.png`

## What the README usually explains

Across the existing records, the README tends to cover:

- the core idea or claimed improvement
- which parts differ from the baseline
- the exact command used
- key metrics from logs
- artifact-size accounting
- training volume and runtime
- the list of included files
- whether the run is leaderboard-track or non-record / unlimited-compute

## Role of `submission.json`

`submission.json` acts as the compact metadata payload for a submission. Existing examples include:

- author and GitHub identity
- run name and short blurb
- date
- `val_loss`
- `val_bpb`
- artifact-size fields such as `bytes_total`, `bytes_code`, and sometimes `bytes_model_int8_zlib`
- track metadata for non-record runs

In practice, the README provides the narrative and reproducibility context, while `submission.json` provides the machine-readable summary.

## How train logs are treated

- Logs are committed as concrete evidence of the run.
- The baseline-style submissions usually keep at least one exact log.
- Record claims often include multiple logs to show variance or statistical significance.
- The log is not just incidental output; it is part of the submission evidence package.

## Non-record submissions still matter

The root README explicitly allows non-record submissions, including unlimited-compute runs, as long as they remain interesting and well justified.

This matters for the current setup:

- we do not need to optimize for SOTA immediately
- we can prioritize a clean, reproducible first branch
- negative or partial results can still be useful if documented well

## Current operating policy

- Do not create a formal `records/...` submission folder yet.
- First target: a non-record candidate with clear provenance and reusable documentation.
- Keep the baseline fixed.
- Use `runs/` and `notes/` for iteration history until there is a submission-worthy packet.
