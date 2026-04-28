# Submissions — Agent Protocol

## You are in: THE SUBMISSION ZONE

This directory handles competition PRs to openai/parameter-golf.
This is the highest-stakes operation in the lab. Slow down. Verify everything.

## Hard stops — check BEFORE doing anything

1. The run must be DONE. Both seeds (444 + 300) complete. Logs saved locally.
2. The model must beat the current LEADER.md score.
3. You must be on TEST_LAB (not in the middle of an experiment).
4. A merged PR MUST NEVER be touched again. Check `gh pr list --repo openai/parameter-golf` first.

## The only workflow

```
bash submissions/validate.sh records/track_10min_16mb/<records_dir>/
  ↓ all checks pass
git checkout -b submission/<name>
git add records/track_10min_16mb/<records_dir>/
git commit -m "Add <Name> submission — <BPB> BPB, <size>MB"
git push fork1 submission/<name>
  ↓ verify on https://github.com/newjordan/parameter-golf-1/branches
gh pr create --repo openai/parameter-golf --head "newjordan:submission/<name>" ...
```

## Remotes (memorize this)

| What | Remote | Repo |
|------|--------|------|
| Daily lab work | `origin` | newjordan/parameter-golf |
| Submission branches | `fork1` | newjordan/parameter-golf-1 |
| PR target | `upstream` | openai/parameter-golf |

`origin` NEVER gets submission branches. `fork1` ONLY gets submission branches.

## Required files in records dir (all four, no exceptions)

- `submission.json` — fill from templates/submission_neural.json or submission_crawler.json
- `train_gpt.py` — EXACT file that ran (vault copy for neural; champion leg copy for crawler)
- `train_seed444.log` — full log
- `train_seed300.log` — full log
- `README.md` — results table + reproduce instructions

## submission.json — critical fields

- `bytes_total` must be the MAX across seeds, must be ≤ 16,000,000
- `bytes_code` must match `Code size:` line in training log
- `val_bpb_exact` must match `final_sliding_window_exact val_bpb=` in log
- `date` is the run date, not submission date

## What killed past PRs

- PR #674 (Podracing, world record 1.0461): closed → no logs, no submission.json. Position lost.
- Rascal II initial push: wrong file (records/ copy 103437 bytes, not vault 118521 bytes). Had to resubmit.

## Never

- Touch a PR that's already merged or open (unless explicitly asked)
- Push a submission branch to `origin`
- Submit from TEST_LAB directly
- Skip validate.sh
- Invent a PR body — use templates/pr_body_template.md
