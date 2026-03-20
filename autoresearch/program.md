# Parameter Golf Autoresearch

This repo uses a lightweight adaptation of `karpathy/autoresearch`.

## Goal

Minimize held-out FineWeb validation loss indirectly through the challenge metric `val_bpb`, while respecting:

- 16,000,000 total bytes for compressed weights plus training code
- 10 minute wallclock budget on 8xH100s

## In-scope files

- `train_gpt.py`
- `train_gpt_mlx.py`
- `autoresearch/run_search.py`
- `justfile`

## Search policy

- Prefer lower `val_bpb`
- Reject runs over the artifact limit
- Prefer simple changes that improve architecture or optimizer hyperparameters
- Focus on changes that are plausible under the real 8xH100 budget

## Metric integrity guardrails

- Use full validation (`VAL_EVAL_MAX_SEQS=0` or unset) for any run you intend to compare against leaderboard/baseline scores.
- Truncated validation (`VAL_EVAL_MAX_SEQS>0`) is for fast local smoke loops only.
- On MLX, quantized roundtrip evaluation runs in a follow-up `MLX_VALIDATE_ONLY=1` process after the parent run writes the quantized artifact.
- CUDA baseline behavior is intentionally unchanged; MLX fast-loop controls are scoped to local iteration.

## Search modes

### `random`

Mutate around the current best config or the backend baseline. This is the simplest default search path and stays close to the current best-known point.

### `preset`

Run one named preset, or sample across all preset families for the selected backend. Use this to compare broad architecture families quickly before settling into narrower search.

### `evolution`

Load prior successful trials from `logs/autoresearch/trials/`, rank them by `val_bpb`, keep a bounded top-K pool, and generate new candidates via crossover plus light mutation.

### `code`

Generate dedicated training-script copies under `logs/autoresearch/workbench/` and apply safe textual mutations there. Baseline training files are never edited in place by this mode.

## Search dimensions

Primary knobs:

- `NUM_LAYERS`
- `MODEL_DIM`
- `NUM_HEADS`
- `NUM_KV_HEADS`
- `MLP_MULT`
- `TRAIN_SEQ_LEN`
- `TRAIN_BATCH_TOKENS`
- `TIED_EMBED_LR`
- `MATRIX_LR`
- `SCALAR_LR`
- `MUON_MOMENTUM`
- `MUON_BACKEND_STEPS`
- `QK_GAIN_INIT`
- `LOGIT_SOFTCAP`
- `WARMDOWN_ITERS`
- `TIED_EMBED_INIT_STD`

## Operating mode

For local iteration on Apple Silicon, use `--backend mlx` and start with `preset` or `random`.

Once a few successful MLX trials exist, `evolution` becomes useful for recombining prior good configs.

For challenge-oriented search, use `--backend cuda` and scale `--nproc` to the target machine. CUDA `preset` and `random` are good for validating families remotely; CUDA `evolution` and `code` are the more aggressive search modes.

## Outputs

Every trial appends to `logs/autoresearch/results.tsv` and the current best run is mirrored to `logs/autoresearch/best_config.json`.

Additional artifacts:

- `logs/autoresearch/trials/*.json`: full per-trial metadata used for resume, evolution, and provenance inspection
- `logs/autoresearch/workbench/*`: generated candidate scripts for code-mutation runs
- `logs/autoresearch/*.log`: stdout/stderr captured for each run

## Suggested command flow

Local MLX:

```bash
just autoresearch-preset-mlx 5 1337 micro_smoke
just autoresearch-preset-mlx 5 1337 small_fast
just autoresearch-mlx 5 1337
just autoresearch-evolution-mlx 5 1337 6
```

Remote CUDA:

```bash
just autoresearch-preset-cuda 5 1 1337 depth_first
just autoresearch-cuda 5 1 1337
just autoresearch-evolution-cuda 5 1 1337 6
just autoresearch-code-cuda 5 1 1337 plain_logits
```
