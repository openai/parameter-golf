# Spectral Flood Walk v0

This record folder is the first RunPod-ready packaging for the local `Spectral Flood Walk` experiments.

It is intentionally aimed at the **non-record** track right now. The goal is to make the retrieval-based prototype runnable in the same operational shape as accepted submissions:

- run from inside a record folder
- produce a `train.log` automatically
- emit a machine-readable `result.json`
- export a compact `seed_pool.npz`
- export a quantized `model_int8.npz`
- keep the experiment configuration attached to the record folder

## What this run is

`train_gpt.py` in this folder is a thin wrapper around the root [`spectral_flood_walk_v0.py`](../../../spectral_flood_walk_v0.py). The root script contains the actual model and training loop:

- 8 affine low-rank recurrent experts
- prefix query path with STE fake-quant int8 queries/keys
- streaming retained seed bank
- optional `runtime` bank mode that stores expanded retrieval states directly on GPU
- read-mass aware retention
- retrieval dropout / warmup schedule
- lightweight CUDA memory telemetry in the training/eval logs
- fixed-pool retrieval eval comparison

The wrapper exists so we can use the standard Parameter Golf workflow on RunPod without re-plumbing commands every time.

## RunPod Preflight

Before spending real credits on a pod, run the lightweight hardware screen in
[`docs/runpod-preflight.md`](../../../docs/runpod-preflight.md). It captures the
`30`-second benchmark from Parameter Golf discussion `#743`, the quick H100 SXM
thresholds, and the keep-vs-reroll workflow we want to use before step 4.

For convenience, this folder also includes a wrapper script:

```bash
./runpod_preflight.sh
```

It writes the raw output to `runs/<timestamp>_preflight.log`.

## Fast Workflow

These wrappers are the quickest way to iterate once the pod is up:

```bash
# fast DDP / artifact smoke test
./runpod_smoke.sh

# first serious run with the current default H100-oriented profile
./runpod_full.sh

# three full seeds back-to-back (defaults: 1337, 42, 2025)
./runpod_three_seeds.sh
```

Each wrapper creates a fresh run directory under `runs/` and writes:

- `train.log`
- `result.json`
- `seed_pool.npz`
- `model_int8.npz`
- `command.sh`
- `notes.txt`

The run wrappers intentionally override the root-level artifact paths so repeated
experiments do not clobber each other. Once you decide which run is worth
freezing into the record folder, promote it with:

```bash
./promote_run.sh runs/<timestamp>_<profile>_seed<seed>
```

## Step 4 Command

From a RunPod shell in `/workspace/parameter-golf`:

```bash
cd records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v0
./runpod_full.sh
```

This is the recommended serious-run entrypoint because it applies the current
H100-oriented dimension overrides, bank mode defaults, artifact paths, and log
layout for you. The manual `torchrun train_gpt.py ...` path is still useful for
debugging, but it is easier to accidentally run with smoke-scale dimensions.

`./runpod_full.sh` writes:

- `train.log`
- `result.json`
- `seed_pool.npz`
- `model_int8.npz`

automatically into this folder.

The shell wrappers now default to:

- `--bank-store-mode runtime`
- `--bank-runtime-dtype fp16`
- `--weight-decay 0.01`

so pod runs keep expanded retrieval states in GPU memory instead of only compact
latent codes. When running on CUDA, the periodic `[train]` log lines also append
VRAM telemetry and the current resident bank estimate.

To turn on the local `v1` behavior, set:

```bash
SFW_EVAL_ONLINE_APPEND=true
SFW_EVAL_APPEND_WRITES_PER_BATCH=<n>
```

That causes the retrieval-enabled eval pass to clone the trained seed bank,
append newly scored entries after each chunk, and report the initial/final bank
sizes plus appended-entry count in `result.json`.

To resume from a previously exported seed pool, set:

```bash
SFW_SEED_POOL_LOAD_PATH=/absolute/path/to/seed_pool.npz
```

The wrapper will pass that through to `train_gpt.py`, preload the bank, and
continue training/eval from the saved entries.

## Notes

- The current distributed path uses DDP for the model but still keeps a **local retained bank per rank**. That is enough for non-record experimentation and log generation, but it is not yet the final multi-GPU retrieval design.
- `result.json` includes `eval_with_retrieval`, `eval_without_retrieval`, `val_bpb`, delta metrics, training throughput, bank summary information, train/eval CUDA memory stats when available, and the exported seed/model artifact paths and sizes.
- When online append is enabled, `result.json` also records `bank_initial_size`, `bank_final_size`, and `appended_entries` for the retrieval eval.
- If you want a smaller first RunPod smoke test, reduce `train_steps`, `train_tokens`, and `eval_samples`.

## Expected Next Step

Once we have a real 8xH100 log here, the next workflow step is:

1. inspect `train.log` and `result.json`
2. decide whether this is worth freezing into a full standalone record script
3. if yes, add `submission.json` and polish the record package
