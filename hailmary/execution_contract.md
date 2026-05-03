# Hailmary Execution Contract

This file defines the active executable claim for `hailmary`.

## Active Base

`hailmary` now runs on the current executable strong local base:

- `VOCAB_SIZE=8192`
- `DATA_PATH=./data/datasets/fineweb10B_sp8192`
- `TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model`
- `XSA_LAST_N=11`
- `QK_GAIN_INIT=5.25`
- `VE_ENABLED=1`, `VE_DIM=128`, `VE_LAYERS=9,10`
- `MUON_WD=0.095`, `ADAM_WD=0.04`
- current local root `train_gpt.py` plus runtime patches

This is a strong executable moonshot base, not a full recurrence/MuonEq-R frontier port.

## Active Tournament Claim

The active tournament is meant to compare implemented broken-invariant mechanisms on
that executable base. It is not meant to claim that all deferred frontier-rebase
ideas are already runnable.

## Active Primary Packs

These are the packs that define the current executable stage thesis:

- `ttt_family`
- `phase_split`
- `checkpoint_selection`
- `staged_curriculum`

## Active Support Packs

These are runnable but subordinate:

- `alternating_objective`

## Active Export Anchors

These are still part of the active tournament finalist wave as fixed export
references:

- `G1`
- `G2`

## Legacy / Deferred Packs

These are preserved as references or deferred rebuild lanes and are not part of the
active tournament thesis:

- `moonshot_core`
- `moonshot_second_wave`
- `moonshot_geometry`
- `moonshot_throughput`
- `parameter_family_split`
- `context_stage`

See [rebase_hypotheses.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hailmary/rebase_hypotheses.md) for the larger frontier-rebase ideas that still need dedicated code.
