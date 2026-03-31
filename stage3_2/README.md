# Stage 3.2

State-conditioned hyperparameter control for `parameter-golf`.

This stage is a shift from static patches to bounded dynamic policies.

The core idea is:

- observe cheap training-state signals
- adjust a small set of high-leverage controls
- evolve those policies under a bounded DSL

This exists because many promising mechanisms are phase-misaligned:

- helpful late, harmful early
- helpful for deployed score, not raw loss
- helpful for one tensor family, harmful for another

Static patches do not express that well. A bounded controller does.

Files:

- [bar_assessment.md]( nanoevolve/pgolf/parameter-golf/stage3_2/bar_assessment.md)
  - which controller families clear the idea bar and the expected lift range for each
- [controller_dsl.md]( nanoevolve/pgolf/parameter-golf/stage3_2/controller_dsl.md)
  - bounded policy representation: signals, actions, gates, transitions, mutation targets
- [hypotheses.md]( nanoevolve/pgolf/parameter-golf/stage3_2/hypotheses.md)
  - dynamic-policy hypothesis families and expected signal channels
- [search_plan.md]( nanoevolve/pgolf/parameter-golf/stage3_2/search_plan.md)
  - staged tournament design for evolving dynamic controllers
- [patches.py]( nanoevolve/pgolf/parameter-golf/stage3_2/patches.py)
  - runtime state-controller patch against the copied strong base
- [run_configs.json]( nanoevolve/pgolf/parameter-golf/stage3_2/run_configs.json)
  - first runnable controller slate
- [run_strategy.py]( nanoevolve/pgolf/parameter-golf/stage3_2/run_strategy.py)
  - entrypoint for `sanity -> screen -> decision -> champion_8x`
- [controller_dsl.py]( nanoevolve/pgolf/parameter-golf/stage3_2/controller_dsl.py)
  - actual Python schema and canonicalization for controller specs
- [controller_library.py]( nanoevolve/pgolf/parameter-golf/stage3_2/controller_library.py)
  - seed controllers and candidate metadata
- [controller_mutations.py]( nanoevolve/pgolf/parameter-golf/stage3_2/controller_mutations.py)
  - bounded structural, wiring, and numeric mutations
- [evolve_stage3_2.py]( nanoevolve/pgolf/parameter-golf/stage3_2/evolve_stage3_2.py)
  - generation builder and seed/child config emitter
- [run_evolution.py]( nanoevolve/pgolf/parameter-golf/stage3_2/run_evolution.py)
  - wrapper for the separate evolutionary path

Working rule:

- `stage2_1` searched static frontier-aligned stacks
- `hailmary` searched larger broken-invariant static/process hypotheses
- `stage3_2` searches bounded state-conditioned control policies

First runnable lead families:

- `H201` late deploy gate
- `H202` best deployed-state selection
- `H204` family-split warmdown
- `H205` alternating objective pulses

Support variants in the first pack:

- `H202B` raw-scored checkpoint selection
- `H206` systems-aware late deploy gate
