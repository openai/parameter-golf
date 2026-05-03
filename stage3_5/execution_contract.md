# Stage 3.5 Execution Contract

This file defines the active executable claim for `stage3_5`.

## Active Base

`stage3_5` runs on the current executable strong local base:

- `VOCAB_SIZE=8192`
- `DATA_PATH=./data/datasets/fineweb10B_sp8192`
- `TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model`
- `XSA_LAST_N=11`
- `QK_GAIN_INIT=5.25`
- `VE_ENABLED=1`, `VE_DIM=128`, `VE_LAYERS=9,10`
- no recurrence or MuonEq-R in the active copied base script

This is a strong local executable trunk, not a full frontier-recurrence port.

## Active Mechanism

The runnable mechanism is exactly one patch family:

- `event_branch_tournament`

Every active candidate in [run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_5/run_configs.json) must reduce to a different branch trigger plus branch-program specification over that patch.

## Active Hypothesis Families

These are the families the code actually runs:

- `H501` pre-quant TTT tri portfolio
- `H502` TTT breadth-vs-depth duel
- `H503` plateau-gated aggressive swing
- `H504` export-state style tournament
- `H505` TTT-vs-recurrent deploy duel
- `H506` failsafe event tri

## Deferred Families

These are not active executable claims in this stage:

- ETLB branches
- recurrence-aware branch programs on a true recurrent trunk
- SP4096/SP8192 architecture co-search

Those belong to [rebase_hypotheses.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_5/rebase_hypotheses.md), not to the current runnable tournament.
