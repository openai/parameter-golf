# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OpenAI's Parameter Golf challenge: train the best language model fitting in a 16MB artifact (16,000,000 bytes) within 10 minutes on 8xH100s, scored by bits-per-byte (val_bpb) on FineWeb validation.

## Commands

```bash
# Setup
uv sync --extra mlx          # Apple Silicon
uv sync --extra cuda          # CUDA machines

# Download data (pass shard count: 1 for smoke, 10 for full)
just download-data 10

# Quick local smoke test
just mlx-smoke

# Full MLX training
just mlx-train mlx_run 2000 524288 0 524288

# CUDA training (single GPU)
just torch-train baseline_sp1024 1

# Tests
just test
# or: uv run python3 -m unittest discover -s tests -p "test_*.py"
# Single test: uv run python3 -m unittest tests.test_run_search.RunSearchPersistenceTests.test_make_result_keeps_description_outside_config

# Autoresearch (automated hyperparameter search)
just autoresearch-mlx 5 1337                      # random search
just autoresearch-preset-mlx 5 1337 small_fast     # preset search
just autoresearch-evolution-mlx 5 1337 6           # evolutionary search
just autoresearch-code-mlx 5 1337 gelu_mlp         # code mutation search
just autoresearch-resume mlx 5 1 1337              # resume from best
```

## Architecture

**Two training backends** with parallel implementations:
- `train_gpt.py` — PyTorch/CUDA, multi-GPU via torchrun. The canonical submission script.
- `train_gpt_mlx.py` — MLX for Apple Silicon local iteration. Mirror of the same model architecture.

Both scripts are capped at 1500 lines max. They define a `Hyperparameters` class driven entirely by environment variables (e.g. `NUM_LAYERS`, `MODEL_DIM`, `ITERATIONS`, `SEED`).

**Autoresearch system** (`autoresearch/run_search.py`):
- Launches training subprocess with mutated env vars, parses stdout for `val_bpb` and model size via regex.
- Four search modes: `random`, `preset`, `evolution`, `code`.
- State persisted in `logs/autoresearch/`: `results.tsv`, `best_config.json`, `trials/*.json`, `workbench/` (code mutation scripts).
- Key constraint: artifact must pass int8+zlib roundtrip eval under 16MB.

**Scoring pipeline**: train → quantize to int8 → zlib compress → measure total bytes (code + model) → evaluate val_bpb from the compressed checkpoint. The `final_int8_zlib_roundtrip` log line is the authoritative result.

## Key Constraints

- All hyperparameters are set via environment variables, not CLI args.
- The 16MB limit is decimal (16,000,000 bytes), not 16 MiB.
- Validation always runs on the fixed first-50k-document FineWeb val split.
- Data lives in `./data/datasets/fineweb10B_sp1024/` and tokenizer in `./data/tokenizers/`.
- Submissions go in `records/` subdirectories, not by modifying the base scripts.
