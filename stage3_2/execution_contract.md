# Stage 3.2 Execution Contract

This file defines the active executable claim for `stage3_2`.

## Active Base

`stage3_2` runs on the current executable strong local base:

- `VOCAB_SIZE=8192`
- `DATA_PATH=./data/datasets/fineweb10B_sp8192`
- `TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model`
- `XSA_LAST_N=11`
- `QK_GAIN_INIT=5.25`
- `VE_ENABLED=1`, `VE_DIM=128`, `VE_LAYERS=9,10`
- `MUON_WD=0.095`, `ADAM_WD=0.04`
- no recurrence or MuonEq-R in the active copied base script

This is a strong local executable trunk, not a full Era-6 frontier port.

## Active Mechanism

The runnable mechanism is exactly one patch family:

- `state_controller`

Every active candidate in [run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_2/run_configs.json) must reduce to a different controller spec over that patch.

## Active Hypothesis Families

These are the families the code actually runs:

- `H201` late deploy gate
- `H202` best deployed-state selection
- `H204` family-split warmdown
- `H205` alternating objective pulses
- `H207` best-state plus narrow pre-quant TTT
- `H208` best-state plus broader dTTT-style tail

## Deferred Families

These are not active executable claims in this stage:

- recurrence activation control
- recurrence-to-TTT handoff control
- recurrence-quant coupling
- MuonEq-R-specific controls

Those belong to [rebase_hypotheses.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_2/rebase_hypotheses.md), not to the current runnable tournament.
