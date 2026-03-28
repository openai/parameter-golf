# Workspace instructions for AI coding assistants

This repository is an OpenAI challenge workspace for the Parameter Golf model-crafting competition. The primary focus is on training and experimenting with compact language model architectures and compression techniques that fit in a 16MB artifact and can be trained quickly.

## What matters here
- `train_gpt.py` and `train_gpt_mlx.py` are onboarding / baseline training scripts.
- `data/` contains dataset download helpers, tokenizer exports, and data pipeline utilities.
- `records/` contains experiment submissions and run-specific logs.
- `README.md` is the main project documentation and should be the first reference for repository purpose and workflows.

## Primary tasks for this workspace
- Improve or extend training workflows in `train_gpt.py` or `train_gpt_mlx.py` without making them unreadably complex.
- Add new experiment recipes or benchmark runs under `records/`.
- Update data download/support scripts under `data/` when needed.
- Preserve the challenge convention: these scripts are not meant to be SOTA codebases, just good starting points.

## Key conventions
- Keep `train_gpt.py` and `train_gpt_mlx.py` readable and short. The repository explicitly prefers these starter scripts to remain under ~1500 lines.
- Use environment variables for configuration rather than hard-coding new values when possible.
- Avoid changing existing historical run data in `records/` unless the user explicitly requests editing a specific experiment.
- Prefer small, targeted changes over broad refactors.

## How to run
Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

Download the cached FineWeb dataset:
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

Run the baseline training script:
```bash
python3 train_gpt.py
```

Run the MLX-compatible version:
```bash
python3 train_gpt_mlx.py
```

Override defaults using environment variables, for example:
```bash
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 python3 train_gpt_mlx.py
```

## Important files
- `README.md` — challenge overview, leaderboard, and usage notes.
- `requirements.txt` — Python dependencies.
- `data/README.md` — dataset download and tokenizer workflow.
- `train_gpt.py` — single-process baseline training script.
- `train_gpt_mlx.py` — Apple Silicon / MLX training path.
- `records/` — experiment histories and reference runs.

## When assisting in this repo
- Link to existing docs rather than duplicating challenge details.
- If asked to add a new experiment, suggest a new `records/` subdirectory and a short `README.md` for the run.
- If asked to add instructions, keep them specific to model training, data preparation, and experiment recording.
- No explicit automated tests are present; use code inspection and repository docs for guidance.
