# Local MLX Run Discipline

## What NOT to touch
For any run meant to be comparable to prior results, do NOT change these unless explicitly testing them:

- Python path: always `.venv/bin/python3`
- `TRAIN_BATCH_TOKENS`
- `GRAD_ACCUM_STEPS`
- `MLX_MAX_MICROBATCH_TOKENS`
- `WARMUP_STEPS`
- `VAL_BATCH_SIZE`
- `MAX_WALLCLOCK_SECONDS`
- `VAL_LOSS_EVERY`

If you change those casually, the run is no longer comparable and MLX behavior can change a lot.

## Why the current branch feels "hung"
The current best branch is compile-heavy. MoD-lite token routing in the MLP path means compile/warmup is much slower than earlier branches. The 180s cap only applies to measured training, not the up-front warmup/compile.

- Long time at `warmup_step:1/20` is **expected**
- That does **not** mean training is broken
- The run can still be healthy even if nothing interesting appears for a while

## How to verify a run is alive
```bash
ps aux | rg train_gpt_mlx.py
tail -n 40 logs/<RUN_ID>.txt
```

Healthy signs:
- Config block printed in log
- `warmup_step:1/20` eventually appears
- Process exists in `ps`

That is enough. Do NOT start improvising fixes mid-run.

## Two modes only

### Benchmark mode
Used for numbers we compare against prior experiments.
Do NOT change infrastructure knobs. Use the exact known-good command template.

### Debug mode
Used only to test whether something launches. Fine to reduce:
- `WARMUP_STEPS=1`
- `MAX_WALLCLOCK_SECONDS=30`
- maybe `VAL_MAX_TOKENS`

But **never compare debug runs to benchmark runs**.

## Known-good benchmark template
For current best branch comparisons, start from exactly this and only change the variable under test:

```bash
DEPTH_SHARE_MODE=cycle DEPTH_UNIQUE_LAYERS=3 DEPTH_SHARE_HEAVY_ONLY=1 \
MUON_WEIGHT_DECAY=0.01 SMEARGATE=1 SMEARGATE_INIT=-3.0 \
MOD_KEEP=0.75 MOD_CORE=1 ATTNRES_MODE=none \
MAX_WALLCLOCK_SECONDS=180 VAL_LOSS_EVERY=0 SEED=1337 \
RUN_ID=<new_run_id> \
.venv/bin/python3 train_gpt_mlx.py
```

If testing one thing, change one thing.

## What to NEVER do
- Don't switch Python binaries
- Don't tweak batch/chunk settings "to help"
- Don't interrupt because warmup feels long
- Don't stack multiple accidental changes in one run
- Don't trust the terminal pane more than the log file

## Simple rule
If a run looks wrong, first move:
1. `inspect logs/<RUN_ID>.txt`
2. `inspect ps`
3. Ask for interpretation

NOT "change settings until it works."
