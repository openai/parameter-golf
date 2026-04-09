# Record: SP8192 + Pre-Quant TTT (QK 5.25, 8ep, freeze-1) — val_bpb 1.0787 (3-seed mean)

**3-seed mean sliding val_bpb:** `1.07873723` (std `0.00049363`)
**3-seed mean roundtrip val_bpb:** `1.09258717` (std `0.00054369`)

Hardware: `8xH100 SXM` | Train cap target: `595s` | Eval: sliding window `stride=64`

## What changed

This package uses the same fixed-predictor lane as the prior SP8192 pre-quant TTT stack, with tuned settings from our April 8 sweep:

- `QK_GAIN_INIT=5.25`
- `TTT_ENABLED=1` with `TTT_EPOCHS=8`, `TTT_LR=0.00045`, `TTT_FREEZE_BLOCKS=1`
- same SP8192 + recurrence + GPTQ pipeline

No tokenizer/dataset modifications, no eval-time adaptation, no SLOT/ngram overlays.

## Seed Results

| Seed | sliding val_bpb | roundtrip val_bpb | train_s | eval_s | bytes_total |
|------|----------------:|------------------:|--------:|-------:|------------:|
| 42 | 1.07913183 | 1.09299539 | 595.162 | 74.678 | 15171524 |
| 1337 | 1.07804121 | 1.09181877 | 595.086 | 74.663 | 15163267 |
| 2025 | 1.07903865 | 1.09294735 | 595.162 | 74.560 | 15188203 |
| **Mean** | **1.07873723** | **1.09258717** | - | - | - |
| **Std** | **0.00049363** | **0.00054369** | - | - | - |

## Sweep Provenance

- best single-seed sweep run (`runB_seed1337`): `1.07765960`
- confirmation seeds in this package: `42, 1337, 2025`
- raw sweep table included in `runs.csv`

## Compliance Notes

- score computed from `final_int6_sliding_window_exact`
- roundtrip reported from `final_int6_roundtrip_exact`
- train capped by wallclock (`stopping_early` line in each log)
- artifact size from `Total submission size int6+brotli+byteshuffle`
