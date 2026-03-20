# Purpose
Repo-local working notes for participating in `openai/parameter-golf` from this checkout.

## Scope
- Keep diffs tight and reviewable.
- Prefer adding local helper docs/scripts over modifying upstream training code unless we are intentionally changing model behavior.
- Competitive or experimental submissions should live in a new folder under `records/`, matching the upstream submission rules.

## Repo Patterns
- `train_gpt.py` is the main CUDA/RunPod path.
- `train_gpt_mlx.py` is the Apple Silicon local-iteration path.
- `data/cached_challenge_fineweb.py` is the canonical dataset downloader.
- `records/` contains concrete submission examples and is the best template for eventual PR structure.

## Local Setup Conventions
- Use `.venv` for the Python environment in this repo.
- On Apple Silicon, install `mlx` in addition to `requirements.txt`.
- Keep downloaded data under `data/datasets/` and tokenizers under `data/tokenizers/`, following upstream defaults.
- See `SETUP.md` for the current local and RunPod bootstrap flow.
- Keep `EXPERIMENT_TRACKER.md` up to date after every meaningful experiment, evaluation, infra change, or decision.
- Treat `EXPERIMENT_TRACKER.md` as the repo-local memory for hypotheses, run commands, metrics, regressions, and next actions.

## Collaboration Workflow
- We will iterate locally together, then you will execute remote runs on RunPod.
- Keep code sync between local and RunPod via git: local changes go to a branch on `origin`, RunPod pulls that branch.
- Record the branch, commit, command, machine, dataset slice, and key metrics for each run in `EXPERIMENT_TRACKER.md`.
- If a run fails because of environment or infra, log it as part of experiment tracking, not just successful runs.

## Submission Guardrails
- Do not guess challenge rules, artifact limits, model IDs, or evaluation behavior; verify against upstream docs.
- New upstream/core-code improvements belong in top-level scripts only if they stay simple; the best models should stay in `records/`.
- Submission PRs should only add a new folder to the appropriate `records/` track and include the required README, `submission.json`, logs, and runnable training script.

## Safety
- Never commit secrets, SSH private keys, tokens, or private URLs.
- Public keys are fine to reference in docs, but do not inline them into committed files unless explicitly requested.
