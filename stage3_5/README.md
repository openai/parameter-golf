# Stage 3.5

Stage 3.5 is a separate stage from `stage3_2` and `stage3_4`.

As of 2026-04-09, it should be treated as a late exploitation stage on the
**current executable strong local base**, not as a full recurrence-frontier port.

The active execution claim is defined in
[execution_contract.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_5/execution_contract.md).

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
- `execution_contract.md`
- `bar_assessment.md`
- `rebase_hypotheses.md`

## Main Mechanism

The lead patch is `event_branch_tournament`.

It adds:

- event-triggered late branching
- finisher presets via a small DSL
- branch-local export portfolio selection (`raw` vs `ema`)
- branch-local pre-quant TTT programs

## 2026-04-09 Active Scope

The branch-tournament idea still makes sense, but the active executable stage is the
strong local 8192-tokenizer trunk plus branch programs. The larger frontier-rebase
ideas remain deferred:

- `SP4096/SP8192`
- `full GPTQ + SDClip + GPTQ embeddings`
- `depth recurrence`
- `MuonEq-R`

The active branch tournament now competes over:

- pre-quant TTT recipes
- export-state choices
- ETLB / eval-time heads
- aggressive late deploy finishers

not over the old `EMA vs late-QAT vs family-split` trio on SP1024.

The current runnable branch programs now compete over:

- narrow pre-quant TTT
- broader freeze-2 TTT
- dTTT-style tail adaptation
- recurrent deploy shaping
- export-state choice inside each branch

See [execution_contract.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_5/execution_contract.md) for the active runnable claim and [rebase_hypotheses.md]( /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_5/rebase_hypotheses.md) for the deferred frontier-rebase branch families.

## Run

```bash
python3 /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_5/run_strategy.py --phase tournament --dry-run
python3 /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_5/run_strategy.py --phase tournament
```
