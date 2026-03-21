# Parameter Golf Autoresearch Program

## Mission
Improve the current `train_gpt.py` in this repository using disciplined, reversible experiments.

Primary objective:
- Minimize `final_int8_ttt_lora` `val_bpb` when test-time training is enabled.

Secondary objectives:
- Keep `final_int8_zlib_roundtrip_exact` competitive.
- Keep total submission size under `16_000_000` bytes.
- Keep evaluation cost reasonable enough that improvements are not purely bought with excessive eval time.

## Current State
- The active trainer is the current root `train_gpt.py`, not the naive baseline copy under `records/`.
- The current trainer already includes LoRA-based test-time training.
- A smoke test on a Runpod `RTX 5090` worked end to end:
  - training completed under a short wallclock cap
  - static quantized eval completed
  - TTT eval completed
- Main observation from the smoke test:
  - training and static eval are cheap
  - TTT eval is the dominant cost

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
For now, use one long-lived worker with a persistent volume.

Why:
- dataset downloads are non-trivial to repeat
- Python dependency setup is slow enough to be annoying
- a persistent worker keeps the feedback loop tight

Recommended pod shape:
- `1x RTX 5090`
- SSH enabled
- no Jupyter
- no volume encryption unless specifically needed
- persistent volume attached at `/workspace`

Use a fresh pod only when:
- the current worker is broken
- we want a clean environment reproduction
- we want a second worker for parallel experiments

## Experiment Discipline
Each experiment must have:
- a short hypothesis
- one family label
- one main change
- one run command
- one judgment

Do not make large stack changes without intermediate evidence.

Preferred experiment families:
- `ttt_scope`
- `ttt_budget`
- `ttt_optimizer`
- `base_model_tradeoff`
- `quantization`
- `batch_size`
- `schedule`

## Current Best Hypothesis Direction
The best next search direction is likely:

- preserve most of TTT's gain
- while cutting TTT evaluation cost significantly

Concrete sub-hypotheses:
- adapt fewer layers
- adapt only `lm_head`
- adapt only late blocks
- reduce LoRA rank
- increase chunk size or reduce update frequency
- adapt selectively instead of uniformly

## First Experiment Queue
Start with these, in order:

1. `ttt_scope/lm_head_only`
- Hypothesis: most TTT gain comes from local output correction, not full Q/V adaptation.

2. `ttt_scope/late_blocks_only`
- Hypothesis: adapting only late layers keeps most of the gain at lower eval cost.

3. `ttt_budget/rank_4`
- Hypothesis: LoRA rank `4` preserves enough adaptation quality while reducing compute.

4. `ttt_budget/chunk_512`
- Hypothesis: larger chunk size lowers TTT update overhead with acceptable quality loss.

5. `ttt_budget/batch_tuning`
- Hypothesis: TTT batch sizing can improve throughput on the 5090 without changing model quality.

6. `base_model_tradeoff/smaller_base_more_ttt`
- Hypothesis: a slightly weaker base model may win overall if it allows stronger or cheaper TTT.

## Standard Run Modes
Use three run modes:

### Smoke
Purpose:
- verify code path
- catch crashes
- confirm logging / parsing

Suggested shape:
- `MAX_WALLCLOCK_SECONDS=30-60`
- reduced `TRAIN_BATCH_TOKENS`
- `VAL_LOSS_EVERY=0`

### Search
Purpose:
- compare candidate hypotheses

Suggested shape:
- `MAX_WALLCLOCK_SECONDS=300`
- enough data shards to avoid obviously degenerate training

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
- `final_int8_ttt_lora val_bpb`
- TTT eval time
- static eval time
- total submission size
- success / failure / invalid

## Keep / Reject Policy
Promote a run only if:
- it completes successfully
- artifact size stays under the cap
- target metric improves
- eval-time growth is acceptable relative to the gain

Archive but do not promote if:
- the result is interesting but too slow
- it improves one metric while regressing another

Reject if:
- crash
- missing final metrics
- invalid size
- improvement too small to justify complexity

## Notes For Future Sessions
- Always inspect the current root `train_gpt.py` before starting a new branch of experiments.
- Prefer modifying one mechanism at a time.
- Treat the remote worker as disposable unless a persistent volume is attached.
- Avoid mixing rule-sensitive ideas with ordinary model-improvement experiments in the same branch.
