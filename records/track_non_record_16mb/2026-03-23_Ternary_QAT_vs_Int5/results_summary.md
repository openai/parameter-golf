# Results Summary

## Environment

- Machine: 1x NVIDIA GeForce RTX 2060 Max-Q (6 GB)
- Dataset: FineWeb-based proxy slice with `1,048,576` train tokens and `262,144` held-out tokens
- Runtime knobs: `COMPUTE_DTYPE=fp32`, `USE_TORCH_COMPILE=0`, `USE_FUSED_OPTIM=0`
- Code size: `62,303` bytes from `train_gpt.py`
- Proxy generator: `make_proxy_dataset.py`

## Phase 0 Sanity Check

The copied top-1 script was isolated into this folder and run locally against the FineWeb-based proxy dataset. A direct fp16 smoke attempt diverged to `NaN`; switching to fp32 plus safer local learning rates produced a stable end-to-end baseline:

- `phase0_baseline_smoke_stable`: `val_bpb=4.10415833`, quantized artifact `9,229,835` bytes

This is only a pipeline sanity check. It is not comparable to the published `1.1428` leaderboard result because the hardware, proxy split, and hyperparameters differ.

## 3-Seed Smoke Matrix

| Group | Config | Mean val_bpb | Std | Mean quantized bytes | Mean zero frac |
|---|---|---:|---:|---:|---:|
| Baseline small int5 | `QUANT_MODE=int5`, 3L, d=256 | 4.055352 | 0.001231 | 1,330,341 | n/a |
| Full ternary | `QUANT_MODE=ternary`, 3L, d=256 | 4.059598 | 0.001478 | 339,876 | 0.566 |
| Mixed ternary | `QUANT_MODE=mixed`, 3L, d=256 | 4.055295 | 0.001233 | 1,177,475 | 0.655 |
| Deep ternary14 | `QUANT_MODE=ternary`, 14L, d=256 | 4.088843 | 0.000453 | 1,099,475 | 0.568 |

Add `62,303` bytes of code to estimate total submission size for these smoke runs.

## Matched Proxy Comparison

These are the most informative local runs because they compare a matched `10L` int5 baseline against mixed ternary at the same shape, then spend the saved byte budget on a larger mixed model.

| Run | Config | Params | Final val_bpb | Total bytes | Delta vs int5 |
|---|---|---:|---:|---:|---:|
| `proxy_int5_10l_seed42_20260323` | `int5`, 10L matched-shape | 25,517,137 | 3.26066687 | 15,406,268 | baseline |
| `proxy_mixed_10l_seed42_20260323` | `mixed`, 10L matched-shape | 25,517,137 | 3.26632616 | 8,705,415 | `+0.00565929` |
| `proxy_mixed_11l640_seed42_20260329` | `mixed`, 11L, `d=640` | 42,630,489 | 3.26090698 | 13,609,639 | `+0.00024011` |

### Key Deltas

1. Same-shape mixed ternary was worse than int5 by about `0.00566` bpb, but saved `6,700,853` bytes.
2. Scaling mixed ternary from the matched `10L` shape to `11L d=640` improved proxy quality by `0.00541918` bpb.
3. The larger mixed run came within `0.00024011` bpb of the matched int5 baseline while still saving `1,796,629` bytes under the `16,000,000` byte cap.
4. Effective compressed model bytes per parameter were about `0.601` for the matched int5 run and `0.318` for the larger mixed run.

## Takeaways

1. Mixed ternary matched the small int5 proxy baseline within noise while reducing model bytes by about 11%.
2. Full ternary reduced model bytes by about 74% versus small int5, but paid about `+0.0043` bpb on the proxy held-out slice.
3. Deeper ternary was clearly worse on quality despite still fitting comfortably in the artifact budget.
4. Same-shape mixed ternary is not enough on its own; the promising path is to use ternary MLP compression to buy a larger model.
5. The larger `11L d=640` mixed run is the strongest result so far. It is effectively tied with the matched int5 proxy baseline on this proxy split while using fewer bytes.
6. Ternary zero fraction stabilized around 56.6% for full ternary and 65.5% for mixed, well below the 80% abort threshold, but mixed runs consistently saturated the worst layer at `1.000` zero fraction in the last MLP projection.

## Per-Seed Runs

| Run | val_bpb | Quantized bytes | Notes |
|---|---:|---:|---|
| `matrix_baseline_small_int5_seed42` | 4.05501795 | 1,332,422 | small int5 baseline |
| `matrix_baseline_small_int5_seed1337` | 4.05699880 | 1,322,682 | small int5 baseline |
| `matrix_baseline_small_int5_seed2024` | 4.05403867 | 1,335,920 | small int5 baseline |
| `matrix_full_ternary_seed42` | 4.05911858 | 341,205 | zero frac 0.566 |
| `matrix_full_ternary_seed1337` | 4.06159870 | 338,830 | zero frac 0.566 |
| `matrix_full_ternary_seed2024` | 4.05807560 | 339,592 | zero frac 0.566 |
| `matrix_mixed_ternary_seed42` | 4.05496354 | 1,173,683 | zero frac 0.655 |
| `matrix_mixed_ternary_seed1337` | 4.05694299 | 1,174,846 | zero frac 0.655 |
| `matrix_mixed_ternary_seed2024` | 4.05397809 | 1,183,897 | zero frac 0.655 |
| `matrix_deep_ternary14_seed42` | 4.08856339 | 1,099,298 | zero frac 0.568 |
| `matrix_deep_ternary14_seed1337` | 4.08848332 | 1,099,008 | zero frac 0.568 |
| `matrix_deep_ternary14_seed2024` | 4.08948193 | 1,100,120 | zero frac 0.568 |
| `proxy_int5_10l_seed42_20260323` | 3.26066687 | 15,343,965 | matched `10L` int5 baseline; total with code `15,406,268` |
| `proxy_mixed_10l_seed42_20260323` | 3.26632616 | 8,643,112 | matched `10L` mixed; total with code `8,705,415` |
| `proxy_mixed_11l640_seed42_20260329` | 3.26090698 | 13,547,336 | larger mixed run; total with code `13,609,639`; zero frac 0.655 |
| `proxy_mixed_11l640_seed1337_20260329` | 3.26105237 | 13,510,406 | larger mixed run; total with code `13,572,709`; zero frac 0.655 |

Raw logs are in `logs/`.
