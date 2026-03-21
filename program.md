# Parameter Golf Autoresearch Program

## Mission
Improve the current root `train_gpt.py` in this repository using disciplined, reversible experiments.

Primary objective:
- Minimize `final_int8_zlib_roundtrip_exact` `val_bpb`.

Constraints:
- Keep total submission size under `16_000_000` bytes.
- Keep training within the intended wallclock budget for the run mode.
- Prefer changes that are likely to transfer from search hardware to stronger reproduction hardware.

## Current State
- The active trainer is the current root `train_gpt.py`, not the frozen copies under `records/`.
- The repo is currently back on the simpler baseline-style trainer, without the TTT-specific focus.
- A smoke test on a Runpod `RTX 5090` proved the remote setup path works:
  - repo clone and venv setup worked
  - FineWeb shard download worked
  - training completed under a short wallclock cap
  - static quantized evaluation completed successfully

## Operating Model
Use a two-role workflow:

1. Controller
- Runs from Cursor / local development.
- Proposes hypotheses.
- Edits `train_gpt.py`.
- Launches or reuses a remote cloud worker.
- Parses logs and records outcomes.

2. Worker
- A single Runpod GPU pod used for execution.
- Holds the repo clone, Python venv, cached dataset, and logs.
- Can be stopped between sessions to save money.

## Cloud Session Strategy
For now, prefer one persistent-volume worker over many short-lived workers.

Why:
- dataset downloads are annoying to repeat
- Python dependency setup takes time
- keeping one warm worker makes the experiment loop much tighter

Recommended pod shape:
- `1x RTX 5090`
- SSH enabled
- no Jupyter
- no volume encryption unless specifically needed
- persistent volume mounted at `/workspace`

Use a fresh pod only when:
- the current worker is broken
- we need a clean reproduction
- we intentionally want a second worker for parallel search

## Experiment Discipline
Each experiment must have:
- a short hypothesis
- one family label
- one main change
- one run command
- one judgment

Do not make large stack changes without intermediate evidence.
Do not mix unrelated ideas in a single run unless we are explicitly testing an interaction.

Preferred experiment families:
- `batch_size`
- `schedule`
- `optimizer`
- `architecture`
- `quantization`
- `precision_budget`
- `context_length`

## Current Best Hypothesis Direction
The most practical next search directions are:

- improve convergence within fixed wallclock
- improve byte efficiency after quantization
- improve parameter allocation inside the same size budget

Concrete sub-hypotheses:
- smaller or larger `TRAIN_BATCH_TOKENS` may improve the update-vs-noise tradeoff
- longer warmdown or tuned Muon momentum may improve the final compressed model
- a slightly different depth/width/MLP allocation may beat the default `9x512` layout
- selective precision or export changes may reduce post-quant degradation
- sequence length changes may improve quality enough to justify any throughput hit

## First Experiment Queue
Start with these, in order:

1. `batch_size/smaller_batch`
- Hypothesis: fewer tokens per step may allow a better optimization tradeoff on single-GPU search hardware.

2. `batch_size/larger_batch`
- Hypothesis: the current baseline may still be underusing stable large-batch training in the search regime.

3. `schedule/longer_warmdown`
- Hypothesis: a longer warmdown improves the final post-quant score more than it costs in earlier-step learning rate.

4. `optimizer/muon_momentum`
- Hypothesis: a slightly higher Muon momentum or longer momentum warmup will improve late-stage convergence.

5. `architecture/mlp_mult_3`
- Hypothesis: a wider MLP may improve parameter efficiency enough to outweigh the extra size after quantization.

6. `architecture/depth_width_trade`
- Hypothesis: a small change in `NUM_LAYERS` and `MODEL_DIM` can improve final `val_bpb` at roughly similar artifact size.

## Standard Run Modes
Use three run modes:

### Smoke
Purpose:
- verify code path
- catch crashes
- confirm logging and parsing

Suggested shape:
- `MAX_WALLCLOCK_SECONDS=30-60`
- reduced `TRAIN_BATCH_TOKENS`
- `VAL_LOSS_EVERY=0`
- `--train-shards 1`

### Search
Purpose:
- compare candidate hypotheses cheaply

Suggested shape:
- `MAX_WALLCLOCK_SECONDS=300`
- enough train shards to avoid obviously misleading results

### Confirm
Purpose:
- re-run promising candidates more seriously

Suggested shape:
- `MAX_WALLCLOCK_SECONDS=600`
- only for candidates that already improved in search mode

## Metrics To Record
For every run, capture:
- run id
- hypothesis
- family
- git diff summary
- GPU type
- wallclock budget
- `final_int8_zlib_roundtrip_exact val_bpb`
- static eval time
- total submission size
- train shard count
- success / failure / invalid

Optional but useful:
- steps reached before wallclock stop
- peak GPU memory
- post-quant size margin to cap

## Keep / Reject Policy
Promote a run only if:
- it completes successfully
- artifact size stays under the cap
- target metric improves
- the gain is meaningful relative to the added complexity

Archive but do not promote if:
- the result is interesting but inconclusive
- it improves one secondary metric while regressing the main one
- it likely needs hardware-specific confirmation

Reject if:
- crash
- missing final metrics
- invalid size
- no meaningful gain

## Notes For Future Sessions
- Always inspect the current root `train_gpt.py` before starting a new branch of experiments.
- Prefer modifying one mechanism at a time.
- Treat the remote worker as disposable unless a persistent volume is attached.
- Keep the first phase focused on static-model improvements before returning to more complex eval-time tricks.
