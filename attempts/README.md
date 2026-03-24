# attempts/

In-progress experiments for the Parameter Golf challenge (OpenAI Model Craft Challenge).

This directory mirrors the structure of `records/track_10min_16mb/` but is intended
for experiments that have not yet been finalized as official submissions.

## Folder structure

Each attempt lives in its own folder named with a date prefix and short description:

```
attempts/
  _template/            # Copy this to start a new attempt
  results.tsv           # Aggregated results across all attempts
  YYYY-MM-DD_ShortDescriptiveName/
    train_gpt.py        # Training script (copy from repo root, then modify)
    submission.json     # Metadata and results (fill in after run)
    README.md           # Hypothesis, changes, results, analysis
    env.sh              # (optional) Environment variable overrides
```

### Naming convention

Use `YYYY-MM-DD_ShortDescriptiveName` with underscores separating words.
This matches the convention in `records/` (e.g. `2026-03-17_NaiveBaseline`,
`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`).

## Workflow

1. **Create folder** -- copy the `_template/` directory:
   ```bash
   cp -r attempts/_template attempts/$(date +%Y-%m-%d)_MyExperiment
   ```

2. **Copy train_gpt.py** -- bring in the current training script:
   ```bash
   cp train_gpt.py attempts/$(date +%Y-%m-%d)_MyExperiment/
   ```

3. **Modify** -- edit `train_gpt.py` and optionally `env.sh` for the changes
   you want to test.  Document your hypothesis in `README.md`.

4. **Submit** -- run on the cluster (sbatch or torchrun). Paste the exact
   command in the attempt's `README.md`.

5. **Collect results** -- fill in `submission.json` and `README.md` with
   metrics.  Append a row to `results.tsv` for easy comparison.

6. **Promote or discard** -- if the attempt beats the current best, copy
   the folder into `records/track_10min_16mb/` and open a PR.  Otherwise,
   update the status to `discarded` and note why.

## Status values

Each attempt tracks a `status` field in `submission.json`:

| Status      | Meaning                                      |
|-------------|----------------------------------------------|
| `pending`   | Created but not yet submitted to the cluster |
| `running`   | Currently executing on the cluster           |
| `complete`  | Run finished, results collected              |
| `failed`    | Run errored or produced invalid results      |
| `discarded` | Results reviewed, not worth pursuing further |
