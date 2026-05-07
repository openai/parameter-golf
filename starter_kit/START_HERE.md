# Parameter Golf Starter Kit

This folder is a low-budget workflow to get from first run to a valid non-record PR.

## 1) Fork + set your remote

From your local repo root:

```bash
git remote rename origin upstream
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/parameter-golf.git
git fetch upstream
git checkout -b exp/first-runs upstream/main
git push -u origin exp/first-runs
```

## 2) On RunPod: first smoke run

Use scripts in `starter_kit/scripts`:

1. `01_runpod_bootstrap.sh`
2. `02_smoke_run.sh`

## 3) Promote to serious run

Run `03_full_run.sh` once smoke logs look healthy.

Optional: run the full 10-run A40 non-record campaign and auto-rank outputs:

```bash
bash starter_kit/scripts/04_non_record_a40_campaign.sh
```

## 4) Prepare a PR-ready records folder

Run:

```bash
python starter_kit/scripts/prepare_submission.py \
  --track non_record_16mb \
  --run-name my_first_non_record \
  --author-name "Your Name" \
  --github-id "your_github" \
  --val-bpb 1.1999
```

Then copy your real train log into the generated folder and edit README details.

## 5) Submission checklist

- Folder only adds one new path under `records/track_non_record_16mb/` or `records/track_10min_16mb/`.
- Includes `README.md`, `submission.json`, `train_gpt.py`, and train log.
- Repro steps are explicit and complete.
- No validation-data leakage or rule violations.
