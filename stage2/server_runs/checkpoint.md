# Stage 2 Checkpoint

## Status

Stage 2 is defined as the first `8xH100` experiment wave built on the public record frontier rather than on our A100 proxy baseline.

## Baselines

There are two baseline roles in Stage 2.

Training trunk baseline:

- [README.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md)
- [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py)

Benchmark target baseline:

Current public record baseline to explain and extend:

- [submission.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/submission.json)
- [train.log](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train.log)
- [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py)

Important caveat for the benchmark target:

- that folder README is stale relative to the record metadata
- use `submission.json` and `train.log` as the source of truth

## Active Stage 2 Doc

- [experiments.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2/experiments.md)
- [strategy.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2/sota_stack_strategy_r1/strategy.md)
- [run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2/sota_stack_strategy_r1/run_configs.json)
- [run_strategy.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2/sota_stack_strategy_r1/run_strategy.py)

## Design Principles

- public SOTA first
- exact reproduction before mutation
- attribution before extension
- one causal question per run
- three-seed confirmation only for real survivors

## Main Questions

1. Which pieces of the public SOTA are essential versus incidental?
2. Which extensions move the strongest clean training trunk?
3. Does our quant-quality win transfer on top of strong `8xH100` baselines?
4. Is eval policy still under-optimized after moving to a stronger checkpoint?
5. Do adaptive Muon, Muon WD, and overtone init compose with the public frontier?
6. Can lower `ms/step` buy enough extra steps to win even when the public stack is already strong?

## Not Doing

- no more Stage 2 decisions based only on A100 proxy values
- no naked depth-only experiment as an early `8xH100` slot
- no premature tokenizer work before export/eval attribution is settled

## Next Actions

1. Run the SOTA-stack `sanity` pack with `R0A`, `R0B`, and `H1-H6`.
2. Run the same SOTA-stack `screen` pack on `1xH100` each.
3. Use `R0A` vs `R0B` spread as the short-horizon noise floor.
4. Promote only the strongest survivor by default to `final_single`.
5. Use `champion_8x` only after the single-GPU finalist looks real.
6. Treat eval-only and export-only moves as deferred lanes after the training-side winner is known.

## Concrete Matrix

Ready scaffold:

- [run_strategy.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2/sota_stack_strategy_r1/run_strategy.py)
- [run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2/sota_stack_strategy_r1/run_configs.json)

Current rule:

- `sota_stack_strategy_r1` is the active direct-run Stage 2 path
- `h100_matrix_r2` remains the generic orchestrator substrate underneath it
