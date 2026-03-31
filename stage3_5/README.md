# Stage 3.5

Stage 3.5 is a separate stage from `stage3_2` and `stage3_4`.

It attacks a bigger false invariant in the same late-stage realm:

- branch time should be chosen in advance
- the late finisher set should be fixed and shallow
- the export state should be predetermined

Instead, this stage:

1. trains a shared trunk
2. triggers branching on training state
3. runs a late finisher tournament from a small DSL
4. evaluates multiple export-state modes inside each branch
5. keeps the best deployed artifact

## Files

- `base_train_gpt.py`
- `patches.py`
- `run_configs.json`
- `orchestrate_stage3_5.py`
- `run_strategy.py`
- `hypotheses.md`
- `bar_assessment.md`

## Main Mechanism

The lead patch is `event_branch_tournament`.

It adds:

- event-triggered late branching
- finisher presets via a small DSL
- branch-local export portfolio selection (`raw` vs `ema`)

## Run

```bash
python3 stage3_5/run_strategy.py --phase all --dry-run
python3 stage3_5/run_strategy.py --phase all
```
