# 10L Int5-MLP + BigramHash(10240) + Delayed PPM K=15

**val_bpb: 1.14174** (mean of 3 seeds, post int5/int6+zstd roundtrip, sliding-window eval stride=32, delayed outside-context-only PPM)

This record keeps the same 10-layer Int5-MLP + BigramHash(10240) base model as `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`, then adds a strictly causal delayed PPM bank at inference time. The PPM bank is only allowed to contain targets from outside the model's current `2048`-token context window.

## Key Result

- **Mean baseline val_bpb:** `1.14299494`
- **Mean delayed-PPM val_bpb:** `1.14173730`
- **Mean improvement:** `-0.00125764` bpb
- **PPM delta std across seeds:** `0.00001979`
- **Paired one-sided p-value (3 seeds):** `4.13e-05`
- **All 3 seeds improved**

## 3-Seed Results

| Seed | Baseline val_bpb | PPM val_bpb | Delta BPB | Total submission size | Valid |
|------|------------------|-------------|-----------|-----------------------|-------|
| 42 | 1.14253746 | 1.14125711 | -0.00128035 | 15,649,761 | yes |
| 1337 | 1.14387335 | 1.14262486 | -0.00124849 | 15,673,878 | yes |
| 2024 | 1.14257402 | 1.14132993 | -0.00124409 | 15,850,793 | yes |
| **Mean** | **1.14299494** | **1.14173730** | **-0.00125764** | | |
| **Std** | **0.00076094** | **0.00076951** | **0.00001979** | | |

## Method

### Outside-Context-Only Delayed PPM

The PPM bank is updated with a delay of `train_seq_len = 2048` tokens. At prediction position `i`, the bank only contains targets from positions `<= i - 2048`, so it cannot exploit anything already visible to the model inside the current sliding-window context.

This preserves the intended use case:

- The transformer handles the local `2048`-token window.
- The delayed PPM bank adds only longer-range repeated-sequence signal.

### Fixed Inference Configuration

- `k_values = [16, 12, 8, 6]`
- `min_confs = [1.0, 1.0, 1.0, 0.95]`
- `min_counts = [1, 1, 1, 1]`
- `boost_k = 15`
- `delay = 2048`
- `bos_id = 1`

`K=15` was selected from an initial seed-42 sweep, then reused unchanged for the validation seeds `1337` and `2024`.

## PPM Bank Stats

These phase-1 stats are identical across seeds because they depend only on the validation tokens and the delayed PPM config:

- Total hits: `631,838`
- Hit rate: `1.019%`
- Direct accuracy: `76.54%`

Per-level hit breakdown:

- `k=16`: `95,920` hits, `91.84%` direct accuracy
- `k=12`: `65,928` hits, `81.75%` direct accuracy
- `k=8`: `194,763` hits, `76.82%` direct accuracy
- `k=6`: `275,227` hits, `69.77%` direct accuracy

## Run Command

```bash
SEED=42 \
RUN_ID=ppm_k15_seed42 \
FINAL_EVAL_PPM=0 \
PPM_SWEEP_K_VALUES='16,12,8,6' \
PPM_SWEEP_MIN_CONFS='1.0,1.0,1.0,0.95' \
PPM_SWEEP_MIN_COUNTS='1,1,1,1' \
PPM_SWEEP_BOOST_KS='15' \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=1337` and `SEED=2024` for the 3-seed validation above.

Files in this folder:

- `train_gpt.py` — delayed-PPM submission entrypoint
- `base_train_gpt.py` — snapshot of the base training script used by the wrapper
- `trie_bench.c` — C helper for delayed trie/PPM bank construction
- `train_seed42.log`, `train_seed1337.log`, `train_seed2024.log` — full training/eval logs
- `submission.json` — leaderboard metadata
