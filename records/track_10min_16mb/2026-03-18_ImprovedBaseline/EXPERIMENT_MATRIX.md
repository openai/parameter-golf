# Improved Baseline Sweep Matrix

This matrix is ordered by expected value under limited compute. Start at the top and stop once one family is clearly winning.

## Baseline

Use the current branch defaults as the anchor:

```bash
RUN_ID=ibl_base \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Tier 1: Fast Ablations

These answer "what is actually carrying?"

| Name | Env overrides | Why |
|---|---|---|
| `abl_no_lora` | `LORA_RANK=0` | Tests whether loop-specific adapters are paying for themselves |
| `abl_no_mtp` | `MTP_HEADS=0 MTP_WEIGHT=0.0` | Tests whether auxiliary prediction is helping |
| `abl_no_ema` | `USE_EMA=0` | Tests whether EMA is helping final eval |
| `abl_no_rec_scale` | `USE_RECURRENCE_SCALES=0` | Tests whether recurrence scales are redundant with LoRA |
| `abl_relu2` | `USE_SWIGLU=0` | Tests whether SwiGLU is actually beating the cheaper ReLU-squared path |

## Tier 2: High-Leverage Sweeps

Run these around the best Tier 1 setting.

| Name | Env overrides |
|---|---|
| `rank8` | `LORA_RANK=8` |
| `rank16` | `LORA_RANK=16` |
| `rank32` | `LORA_RANK=32` |
| `mtp_w005` | `MTP_WEIGHT=0.05` |
| `mtp_w010` | `MTP_WEIGHT=0.10` |
| `mtp_w015` | `MTP_WEIGHT=0.15` |
| `mtp_w025` | `MTP_WEIGHT=0.25` |
| `mtp1` | `MTP_HEADS=1` |
| `mtp2` | `MTP_HEADS=2` |
| `mtp4` | `MTP_HEADS=4` |
| `ema_099_0` | `EMA_DECAY=0.99 EMA_START_STEP=0` |
| `ema_0995_50` | `EMA_DECAY=0.995 EMA_START_STEP=50` |
| `ema_0998_100` | `EMA_DECAY=0.998 EMA_START_STEP=100` |

## Tier 3: Capacity / Layout Sweeps

Only do these after one variant is already clearly beating the baseline.

| Name | Env overrides | Why |
|---|---|---|
| `dim768` | `MODEL_DIM=768` | Spend more of the artifact budget |
| `dim832` | `MODEL_DIM=832` | Aggressive width push if compressed size stays safe |
| `layout_4x3` | `NUM_UNIQUE_LAYERS=4 NUM_RECURRENCE=3` | More effective depth with stronger sharing |
| `layout_3x4` | `NUM_UNIQUE_LAYERS=3 NUM_RECURRENCE=4` | Push recurrence harder |
| `layout_3x5` | `NUM_UNIQUE_LAYERS=3 NUM_RECURRENCE=5` | PR #11-style deeper recurrence regime |

## Recommended Run Order

1. `base`
2. `abl_no_lora`
3. `abl_no_mtp`
4. `abl_no_rec_scale`
5. `abl_no_ema`
6. Best of the above + `rank8`, `rank32`
7. Best LoRA setting + `mtp_w010`, `mtp_w025`
8. Best above + `ema_099_0`, `ema_0998_100`
9. Best above + `dim768`
10. Best above + `layout_4x3`

## What To Log

For each run, keep:

- `final_int8_zlib_roundtrip_exact val_bpb`
- `Total submission size int8+zlib`
- wallclock at stop
- whether it finished cleanly under the cap
