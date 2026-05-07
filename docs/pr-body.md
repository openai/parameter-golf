# PR title

feat: add parameter golf submission prep tooling

# PR body

## What this changes

- adds repo-scoped subagent config for parallel Parameter Golf work
- adds a run helper that can print commands, parse logs, compare runs, and package a completed run into a `records/...` folder
- adds an experiment matrix and submission playbook
- exposes compression-focused trainer knobs in `train_gpt.py`
- adds an opt-in validation cap for local smoke loops without changing default submission behavior

## Why

This branch is not a scored challenge submission. It prepares the repo for one.

The main goal is to make future GPU runs easier to compare and easier to package into a real PR once a winning configuration exists.

## Verified

- `python3 -m py_compile train_gpt.py train_gpt_mlx.py scripts/pg_lab.py`
- parsed the published baseline and 4-hour logs with `scripts/pg_lab.py`
- completed a local MLX smoke run with the opt-in validation cap
- dry-ran `scripts/pg_lab.py prepare-record` into `/tmp` and confirmed it emits `README.md`, `submission.json`, `train.log`, and the copied trainer snapshot

## Notes

- no official leaderboard claim in this PR
- no submission folder under `records/...` yet for a real CUDA run
- `VAL_MAX_SEQS` is for local smoke only and should not be used for a real challenge submission
