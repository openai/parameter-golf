---
title: Submission Playbook
read_when:
  - You have a finished run and want to package it into a PR folder quickly.
  - You are preparing a grant request and need a concrete execution plan.
---

# Submission playbook

## What is ready

The repo now has:

- `scripts/pg_lab.py compare-logs` to rank completed runs by scored post-quant BPB
- `scripts/pg_lab.py prepare-record` to build a `records/...` folder from a finished run log
- compression-first trainer knobs in `train_gpt.py`
- fast smoke validation caps for local sanity checks

## What still needs real compute

Do not submit any run that uses `VAL_MAX_SEQS`.

For a real challenge PR, the winning run still needs:

- the full fixed validation split
- a real CUDA run for `train_gpt.py`
- exact run logs
- a copied trainer snapshot inside the new `records/...` folder

## Prepare a record folder

Example:

```bash
python3 scripts/pg_lab.py prepare-record \
  --log logs/your_real_run.txt \
  --output-root records/track_non_record_16mb \
  --record-name "Sanket Compression Sweep A" \
  --author "Sanket Dongre" \
  --github-id sanky369 \
  --blurb "Compression-first SP-1024 run with post-quant tuning and artifact-aware evaluation." \
  --train-script train_gpt.py \
  --track-label "non-record, 16MB artifact cap" \
  --submission-track non-record-unlimited-compute-16mb \
  --command "RUN_ID=... torchrun --standalone --nproc_per_node=8 train_gpt.py"
```

This writes:

- `README.md`
- `submission.json`
- `train.log`
- the copied trainer snapshot

## Suggested real run order

1. Exact baseline on 1xH100.
2. First 8 sweep runs from `experiments/initial_sweep.csv`.
3. Promote only runs that improve post-quant score or clearly improve speed/bytes.
4. Re-run the top 2 on 8xH100.
5. Package the best result with `prepare-record`.
