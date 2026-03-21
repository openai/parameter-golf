# Parameter Golf Workspace

This repository now has a thin research layer around the existing baseline trainers. The goal is to keep local MLX iteration, remote CUDA baselines, and run comparison clean without turning the baseline scripts into a framework.

## Current Workflow

Core entrypoints:
- `train_gpt_mlx.py`: Apple Silicon / MLX trainer for local iteration.
- `train_gpt.py`: CUDA / torchrun trainer for remote runs and baseline reproduction.
- `data/cached_challenge_fineweb.py`: manifest-driven downloader for published FineWeb shards and tokenizer artifacts.
- `records/`: historical record submissions, each with its own frozen `train_gpt.py`, README, logs, and `submission.json`.

Configuration model:
- Both trainers are primarily configured through environment variables.
- Model shape, optimization hyperparameters, run identifiers, data paths, and wallclock/iteration limits all live in the `Hyperparameters` class inside the trainer script.

Outputs:
- MLX writes logs and model artifacts into `OUT_DIR`.
- CUDA now supports `LOG_DIR` and `ARTIFACT_DIR` so logs and model artifacts can be isolated per run.
- The research launcher writes per-run metadata into `research/results/runs/<timestamp>_<run_name>/`.

## Quick Start

### 1. Install Dependencies

Apple Silicon / MLX local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

CUDA remote setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download the Published SP-1024 Baseline Data

Small local subset:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

Full published baseline subset:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### 3. Run Preflight Checks

MLX local:

```bash
python3 scripts/check_env.py --target mlx
python3 scripts/check_data.py --data-path ./data/datasets/fineweb10B_sp1024 --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model --min-train-shards 1
```

CUDA remote:

```bash
python3 scripts/check_env.py --target cuda
python3 scripts/check_data.py --data-path ./data/datasets/fineweb10B_sp1024 --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model --min-train-shards 1
```

## One-Command Runs

Official local Apple Silicon smoke path:

```bash
python3 research/run.py --preset mlx_smoke
```

Local development preset on Apple Silicon:

```bash
python3 research/run.py --preset local_dev_mlx --run-name local_dev
```

Remote CUDA baseline on one GPU:

```bash
python3 research/run.py --preset cuda_remote_baseline --run-name baseline_1gpu
```

Track-like remote CUDA baseline on 8 GPUs:

```bash
python3 research/run.py --preset cuda_remote_baseline --nproc-per-node 8 --run-name baseline_8gpu
```

Override any trainer env var explicitly:

```bash
python3 research/run.py --preset local_dev_mlx --run-name tied_embed_ablation --set NUM_LAYERS=10 --set MATRIX_LR=0.03
```

## What The Research Layer Records

Each run directory contains:
- `run_spec.json`: resolved preset, command, env, git metadata, and preflight summaries
- `launcher.log`: full console stream from the launched command
- trainer log and trainer artifacts produced by the selected entrypoint
- `result.json`: parsed metrics, wall-clock time, artifact sizes, and submission-budget estimate

Cross-run indexes are written to:
- `research/results/index.jsonl`
- `research/results/index.csv`

Compare runs:

```bash
python3 research/compare_runs.py
```

## Rules-Safe Defaults

The default presets and workspace are designed for:
- baseline reproduction
- small ablations
- architecture and optimizer changes that remain self-contained and reproducible
- compression-aware measurement where the artifact size can be checked directly

Promising rules-safe experiment families:
- baseline-plus: learning-rate, warmdown, seq length, and batch shaping changes
- compression-aware training: quantization-aware or mixed-precision paths that remain reproducible and artifact-compatible
- parameter reuse: tied weights, recurrent block reuse, and other explicit sharing
- small architecture edits: conservative block, skip, or embedding changes
- evaluation hygiene: artifact accounting, exact metric capture, and robust run comparison

Questionable directions are intentionally kept out of the default path. See `research_only/README.md`.
