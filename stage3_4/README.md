# Stage 3.4

Shared-trunk late branching for `parameter-golf`.

This stage is distinct from `stage3_2`.

`stage3_2` changes one trajectory by dynamic control.
`stage3_4` changes the process by refusing to commit to one late trajectory at all.

Core idea:

- train one shared trunk for most of the wallclock
- branch late into several short finisher policies
- export and score each branch
- keep the best deployed branch

Why this exists:

- the late phase is where many ideas interfere
- a single committed late policy may be the wrong abstraction
- the real false invariant may be “one run should choose one finisher in advance”

Files:

- [bar_assessment.md]( nanoevolve/pgolf/parameter-golf/stage3_4/bar_assessment.md)
- [hypotheses.md]( nanoevolve/pgolf/parameter-golf/stage3_4/hypotheses.md)
- [patches.py]( nanoevolve/pgolf/parameter-golf/stage3_4/patches.py)
- [base_train_gpt.py]( nanoevolve/pgolf/parameter-golf/stage3_4/base_train_gpt.py)
- [run_configs.json]( nanoevolve/pgolf/parameter-golf/stage3_4/run_configs.json)
- [orchestrate_stage3_4.py]( nanoevolve/pgolf/parameter-golf/stage3_4/orchestrate_stage3_4.py)
- [run_strategy.py]( nanoevolve/pgolf/parameter-golf/stage3_4/run_strategy.py)
