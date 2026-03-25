# Local Search Playbook

This repository already ships strong baseline trainers, but local iteration on Apple Silicon had two practical gaps:

1. Validation always scanned the full fixed validation split, which is correct for final scoring but too slow for tight local loops.
2. Hyperparameter exploration was entirely manual, so comparing random trials against follow-up refinements was tedious and error-prone.
3. Full-validation confirmations on Macs need larger eval batches than proxy sweeps, otherwise the fixed validation split takes too many tiny batches.

This patch adds a local-first search loop without changing the default challenge path:

- `TRAIN_MAX_SHARDS=0` and `VAL_MAX_SEQS=0` still mean full data and full validation.
- Setting `TRAIN_MAX_SHARDS` lets you restrict training to the first `N` train shards at runtime.
- Setting `VAL_MAX_SEQS` lets you validate on only the first `N` sequences as a proxy metric for local search.
- `tools/local_search.py` automates baseline, random exploration, promotion around the current best run, and optional full-validation confirmation.
- The search harness uses a larger `--full-val-batch-size` for confirmation runs so full validation stays practical on local Apple Silicon machines.
- Use `--space KEY=v1,v2,v3` when you want a tighter search neighborhood around the current frontier instead of the built-in default ranges.
- Use `--space-only` when you want an exact targeted sweep. That freezes every unspecified knob at the values you provided with `--set`.
- Use `--max-param-count` when you start varying architecture knobs so over-budget models are skipped before training.

## Recommended local workflow

1. Install the Apple Silicon path and download one train shard:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

2. Run a quick local search with a proxy validation subset:

```bash
.venv/bin/python tools/local_search.py \
  --python .venv/bin/python \
  --iterations 120 \
  --train-max-shards 1 \
  --val-max-seqs 128 \
  --max-runs 5 \
  --promote-budget 3
```

3. Read the generated report under `experiments/<session>/REPORT.md`.

4. Only opt into full validation when you actually need it, for example:

```bash
.venv/bin/python tools/local_search.py \
  --python .venv/bin/python \
  --train-max-shards 1 \
  --val-max-seqs 128 \
  --full-val-top-k 1
```

On a 16 GB M4-class Mac, full validation is still much slower than proxy sweeps even with a larger confirmation batch.

If you want to search architecture rather than just optimizer knobs, switch to `--space-only` and pin the frontier explicitly:

```bash
.venv/bin/python tools/local_search.py \
  --python .venv/bin/python \
  --iterations 40 \
  --train-max-shards 1 \
  --val-max-seqs 96 \
  --strategy grid \
  --space-only \
  --max-param-count 17059912 \
  --set LOGIT_SOFTCAP=55 \
  --set MATRIX_LR=0.015 \
  --set SCALAR_LR=0.04 \
  --set TIED_EMBED_LR=0.085 \
  --set TIED_EMBED_INIT_STD=0.005 \
  --set QK_GAIN_INIT=1.5 \
  --space NUM_LAYERS=10,11 \
  --space MODEL_DIM=480,512 \
  --space NUM_HEADS=8 \
  --space NUM_KV_HEADS=1,2,4 \
  --space MLP_MULT=2
```

That setup explores only the listed architecture axes and skips candidates whose estimated parameter count exceeds the baseline budget.

If you want a standing cyclic checklist for this workflow, use `TODO.md` in the repository root.

## Search policy

The search script intentionally keeps every run and ranks them by `final_int8_zlib_roundtrip_exact val_bpb`.

- If a promoted follow-up beats the current best run, the search frontier moves to that newer run.
- If a promoted follow-up is worse, the older better run remains the frontier.
- The optional full-validation confirmation reruns only the strongest proxy candidates on the full fixed validation split.
- Successful runs now also record `model_params` and `serialized_model_int8_zlib`, so budget checks stay visible in the report even during local proxy work.

That preserves the history of both "earlier better" and "later worse" runs instead of losing context between experiments.
