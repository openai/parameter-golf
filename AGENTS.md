# Repository Guidelines

## Project Structure & Module Organization
Core training entrypoints live at the repository root: `train_gpt.py` for CUDA and `train_gpt_mlx.py` for Apple Silicon MLX. Automated search code lives in `autoresearch/`, dataset and tokenizer utilities live in `data/`, and helper scripts live in `scripts/`. Keep benchmarkable submissions under `records/track_*/*/` with their own `README.md`, logs, `submission.json`, and runnable training code. `specs/` currently holds planning docs rather than executable tests.

## Build, Test, and Development Commands
Use `uv` for Python environments and `just` as the main task runner.

- `just setup`: install the default MLX local environment.
- `just setup-cuda`: install CUDA dependencies for remote GPU work.
- `just download-data 1` or `just download-data 10`: fetch FineWeb shards plus tokenizer assets.
- `just mlx-smoke`: run a short local sanity check on Apple Silicon.
- `just mlx-train mlx_run 2000 524288 0 524288`: run a longer local MLX training job.
- `just torch-train baseline_sp1024 1`: launch the CUDA baseline with `torchrun`.
- `just autoresearch-mlx 5 1337` or `just autoresearch-cuda 5 1 1337`: run the search harness.

## Coding Style & Naming Conventions
Target Python 3.11+ and follow existing style in the training scripts: 4-space indentation, clear top-level constants, and descriptive snake_case names for variables, functions, and files. Keep new modules focused and avoid adding framework-heavy abstractions to the baseline scripts. Name runs and record folders explicitly, for example `2026-03-17_NaiveBaseline`.

## Testing Guidelines
There is no formal pytest or lint configuration yet. Treat `just mlx-smoke` as the minimum pre-PR validation for local changes, and run the relevant training or `autoresearch` command for the code path you touched. For data or submission changes, verify generated artifacts, logs, and README instructions end to end.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Update README.md` and `Launch snapshot`. Follow that pattern: one concise subject line describing the change. PRs should explain the goal, list the commands you ran, and note any score, dataset, or artifact-size impact. Submission PRs should only add a new folder under the appropriate `records/` track and include all required files described in the main `README.md`.

## Submission & Configuration Notes
Do not rely on network access during evaluation artifacts. Keep counted code self-contained, prefer environment variables over hardcoded machine paths, and document any non-default run settings directly in the submission README.
