# hm2

`hm2` is a bootstrap-to-handoff tournament.

The working hypothesis is:

- some mechanisms are valuable because they accelerate early representation learning
- those same mechanisms can flatten, constrain, or distract the late game
- the right response is not to discard them, but to hand off from them deliberately

`hm2` tests that directly on the strong local `sp8192` base.

It now ships with two base scripts:

- `base_train_gpt.py`
  - the current local strong executable base
- `base_frontier_train_gpt.py`
  - `PR #1416` frontier `train_gpt.py`
  - `SP8192 + pre-quant TTT + SDClip` branch base from `erichroepke/parameter-golf`

The runner supports selecting a base via `HM2_BASE_VARIANT`.
The active pack is still wired to `current_local` because the bootstrap-handoff patch set is currently aligned to that base.

## What It Runs

- `bootstrap_handoff`
  - static bigram prior
  - fixed fade
  - fixed freeze
  - plateau-triggered fade
  - fade plus snapshot selection
  - fade plus pre-quant TTT
- `receiver_mix`
  - plateau fade/freeeze paired with:
    - checkpoint selection
    - pre-quant TTT
    - raw-vs-deployed snapshot choice
    - aggressive snapshot+TTT bundle
- dynamic finalists
  - controls
  - pack winners
  - static anchor
  - aggressive bundle
  - winner plus synthetic snapshot child
  - winner plus synthetic TTT child

## Recording

Every slot records `phase_diagnostics.json` in its run directory.

That file captures:

- early / mid / late loss deltas
- per-bucket step timing
- last in-bucket validation
- handoff events such as trigger and freeze

So this stage is meant to leave behind reusable evidence about:

- what helped early
- what flattened late
- whether the handoff happened at the intended time
- whether the late receiver converted the early gain

## Commands

Dry-run the tournament:

```bash
python3 /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hm2/run_strategy.py --phase tournament --dry-run
```

Run the full tournament:

```bash
python3 /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hm2/run_strategy.py --phase tournament
```

Run one pack only:

```bash
python3 /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hm2/run_strategy.py --phase all --pack bootstrap_handoff
python3 /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hm2/run_strategy.py --phase all --pack receiver_mix
```
