# Autoresearch Expansion Plan

## Goal

Expand the lightweight `autoresearch` integration so it supports all proposed optimization paths for the Parameter Golf challenge:

- architecture presets specialized for the 16 MB / 10 minute target
- evolutionary search seeded from prior best runs
- dedicated-copy code mutation without modifying baseline training files

## Planned Work

### 1. Refactor the runner

Convert `autoresearch/run_search.py` from a single random-search path into explicit modes:

- `random`
- `preset`
- `evolution`
- `code`

Keep the current metric parsing and artifact-limit enforcement as shared plumbing.

### 2. Add architecture presets

Define named preset configs for both `cuda` and `mlx`.

Include challenge-oriented model families such as:

- depth-heavy
- width-heavy
- compact-context / high-throughput

Add CLI support for:

- running a specific preset
- sampling across all presets

### 3. Add evolutionary search

Persist each successful trial as structured JSON.

Load prior successful runs for the current backend.

Maintain a top-K population ranked by `val_bpb`.

Generate new candidates via:

- parent selection
- crossover
- light mutation

Keep resume behavior compatible with the current `best_config.json` workflow.

### 4. Add dedicated-copy code mutation

Never edit `train_gpt.py` or `train_gpt_mlx.py` directly.

For code mode, copy the selected training script into:

- `logs/autoresearch/workbench/`

Apply safe textual mutations only to that copy.

Initial mutation library should target real source patterns such as:

- MLP activation variants
- softcap on/off
- residual-mix simplification
- baked-in default hyperparameter rewrites

### 5. Expand trial metadata

Extend result logging to include:

- `mode`
- `preset`
- `code_mutation`
- parent lineage
- candidate script path

Keep `results.tsv` human-readable.

Store one JSON artifact per trial for resume and evolution support.

### 6. Add `just` commands

Add commands for:

- preset search
- evolutionary search
- code-mutation search

Provide both MLX and CUDA variants.

Keep the current simple commands intact.

### 7. Update docs

Update `README.md` with:

- what each mode does
- recommended order of use
- example commands

Update `autoresearch/program.md` so it reflects the new operating modes and search policy.

### 8. Verify

Run:

```bash
python3 -m py_compile autoresearch/run_search.py
```

Do a dry CLI sanity check for each mode.

Optionally run a very small MLX trial as the first live validation.

## Recommended Execution Order

1. Refactor `run_search.py`
2. Add presets
3. Add evolution
4. Add code-mutation mode
5. Wire `justfile` and docs
6. Sanity-check the new interface

## Deliverables

- expanded `autoresearch/run_search.py`
- updated `justfile`
- updated `README.md`
- updated `autoresearch/program.md`
- persisted per-trial JSON logs for evolutionary resume support
