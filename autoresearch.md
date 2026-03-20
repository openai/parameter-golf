# Autoresearch: modal sweep val_bpb

## Objective
Improve the best `val_bpb` found by a small Modal sweep that exercises multiple architecture ideas from `IDEAS.md`, while watching the submission artifact size and staying aligned with the Parameter Golf constraints.

This is a proxy workload, not the final leaderboard run. The proxy uses small 1xH100 sweeps on `sp1024` with one train shard and short iteration budgets so we can iterate quickly on sweep machinery, ranking, and sweepable model ideas. Keep changes that improve the proxy's best `val_bpb` without blowing past the 16,000,000-byte artifact cap.

## Metrics
- **Primary**: `best_val_bpb` (unitless, lower is better)
- **Secondary**: `best_submission_size_bytes`, `best_compressed_model_bytes`, `best_model_params`, `constraint_ok`, `job_count`

## How to Run
`./autoresearch.sh` — runs a small Modal sweep and prints `METRIC ...` lines.

## Files in Scope
- `modal_app.py` — Modal sweep orchestration, summary extraction, run inspection, sweep dimensions, ranking.
- `IDEAS.md` — backlog and tracking for sweep-ready ideas.
- `README.md` — docs only when the user-facing workflow changes materially.
- `train_gpt.py` — training/model changes that are sweepable and likely to lower `val_bpb` under the artifact cap.
- `autoresearch.sh` — benchmark harness for this loop.
- `autoresearch.checks.sh` — fast correctness checks.

## Off Limits
- `records/` — do not rewrite historical submissions.
- Dataset artifacts under `data/datasets/` and `data/tokenizers/`.
- Any change that depends on network access at challenge evaluation time.

## Constraints
- Artifact must stay under **16,000,000 bytes**.
- Final target is to beat **1.2244 val_bpb**.
- Leaderboard target must evaluate under **10 minutes on 8xH100s**.
- Proxy benchmark should stay small enough for iterative Modal autoresearch.
- Keep the Modal workflow usable from CLI (`train`, `sweep`, `summary`, `log`).
- Fast checks must pass (`python3 -m py_compile ...`).

## Workload
The benchmark runs a small Modal architecture sweep on `sp1024` with one training shard and a short iteration budget. Current default grid:
- `NUM_LAYERS`: `9,12`
- `MODEL_DIM`: `512,640`
- `NUM_HEADS`: `8`
- `NUM_KV_HEADS`: `4`
- `MLP_MULT`: `2`

This gives 4 jobs per sweep and reports the best **constraint-satisfying** run's `val_bpb` plus size metrics from the generated sweep summary.

## What's Been Tried
- Baseline Modal app setup added: data preparation, train/sweep/summary/log commands, and ranking by `val_bpb`.
- Initial benchmark target uses architecture sweeps first because they map directly to the active queue in `IDEAS.md` and are easy to compare across changes.
- Adaptive short-run schedule defaults improved proxy `val_bpb`, but only by allowing the sweep winner to exceed the 16,000,000-byte cap. The benchmark now filters ranking to constraint-satisfying runs only.
- Still to explore: stronger sweep spaces, richer parsing/reporting, schedule sweeps that preserve the cap, and architecture ideas in `train_gpt.py` that improve the constrained sweep frontier.
- External inspiration to test carefully: PR #250 ideas that might transfer without overcomplicating the baseline, especially cosine warm restarts / SGDR for faster loss reduction under fixed wallclock. Avoid copying speculative MoE/PID complexity into the proxy until simpler schedule wins are exhausted.
