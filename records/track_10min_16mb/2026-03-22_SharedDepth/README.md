# Shared-MLP 10L Variant (In Progress)

This directory is a local experiment branch built from
`records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`.

The goal is to test whether shared-depth can help the 10-minute / 16 MB track by
reducing artifact pressure without changing the 10 logical-layer topology.

No validated score is claimed from this directory yet.

## Architecture Change

The model still has 10 logical layers:

- 5 encoder logical layers
- 5 decoder logical layers

Attention and control parameters remain unique per logical layer:

- attention weights
- RMSNorms
- `attn_scale`
- `mlp_scale`
- `resid_mix`
- skip weights

Only the MLP weights are shared.

Current sharing pattern:

- encoder MLP ids: `0,1,2,0,1`
- decoder MLP ids: `0,1,2,0,1`

So the model keeps 10 logical blocks, but uses only 6 unique MLP pools total:

- 3 encoder MLPs
- 3 decoder MLPs

The shared MLP is called inside a single block forward path:

```python
x = block(x, x0, chosen_mlp)
```

This keeps the graph closer to the original baseline than the earlier split
`forward_attn` / `forward_mlp` refactor.

## Current Defaults

- `NUM_LAYERS=10`
- `BIGRAM_VOCAB_SIZE=10240`
- `BIGRAM_DIM=128`
- `SHARED_MLP_ENABLED=1`
- `SHARED_ENCODER_MLP_IDS=0,1,2,0,1`
- `SHARED_DECODER_MLP_IDS=0,1,2,0,1`

This is intentionally the lean shared version. It does not reinvest saved bytes
into more layers or a larger bigram table.

## Export Policy

The current export is set up to spend saved bytes on gentler serialization rather
than added capacity.

- shared MLP pools (`encoder_mlps.*`, `decoder_mlps.*`) export as int6
- ordinary MLP weights still export on the int5 MLP path
- default fp16 keep patterns:
  - `tok_emb`
  - `blocks.8.attn.c_k`
  - `bigram.proj`
- default pruning is disabled with `PRUNE_FRAC=0.0`

## Parameter Count

- March 20 baseline: `25,517,137`
- current shared-MLP variant: `19,225,681`
- saved parameters: `6,291,456`

The earlier reinvested variant with 4 unique MLPs per half and
`BIGRAM_VOCAB_SIZE=16384` was discarded after it lost on BPB, size, and steps.

## Intended Question

This branch is trying to answer:

> Can shared MLP beat the original once the saved bytes are used to reduce export
> damage rather than add model capacity?

If this lean shared version still underperforms, the evidence against shared-depth
in this challenge setup is much stronger.

## Run Command

Run from the repo root:

```bash
RUN_ID=shared3_clean_seed2024 \
SEED=2024 \
NUM_LAYERS=10 \
BIGRAM_VOCAB_SIZE=10240 \
BIGRAM_DIM=128 \
SHARED_MLP_ENABLED=1 \
SHARED_ENCODER_MLP_IDS=0,1,2,0,1 \
SHARED_DECODER_MLP_IDS=0,1,2,0,1 \
FP16_KEEP_NAME_PATTERNS=tok_emb,blocks.8.attn.c_k,bigram.proj \
PRUNE_FRAC=0.0 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/phat-train/train_gpt.py
```
