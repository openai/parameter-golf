# Technical Report: Ternary QAT vs Int5 Proxy Study

## Status

This experiment is no longer at the "does ternary run at all?" stage. The code path is implemented, local proxy evaluation is working end to end, and the current question is narrower: can mixed ternary use the 16 MB artifact budget more efficiently than the current int5-style baseline?

As of `2026-03-31`, the answer from local proxy experiments is:

- Full ternary is not the best next branch.
- Mixed ternary is the only ternary branch that remains competitive.
- A larger mixed model is now effectively tied with the matched int5 baseline on the local FineWeb-based proxy split while using fewer bytes.

This is still a development result, not a leaderboard result. Official FineWeb validation and 8xH100 timing remain untested locally.

## Goal and Constraints

The competition target is a language model artifact under `16,000,000` bytes that trains in at most `600` seconds on `8xH100`, scored by `val_bpb` on the FineWeb validation set.

The local machine does not have the full challenge environment, so this work uses a FineWeb-based proxy dataset for development:

- train tokens: `1,048,576`
- held-out tokens: `262,144`
- tokenizer: `fineweb_1024_bpe.model`
- hardware: `1x RTX 2060 Max-Q (6 GB)`

This proxy is suitable for directional decisions, not for claims about final leaderboard quality.

## Implementation Summary

All experiment changes live in this folder. The repo root script was left alone.

The modified `train_gpt.py` adds:

- `QUANT_MODE=int5|ternary|mixed`
- `TernaryLinear`, using STE-based ternary weights in `{-1, 0, +1}`
- ternary export helpers for packing and roundtrip dequantization
- generic quantized evaluation after export
- local portability flags:
  - `COMPUTE_DTYPE`
  - `USE_TORCH_COMPILE`
  - `USE_FUSED_OPTIM`
- ternary zero-fraction monitoring with an abort threshold

The export path preserves a meaningful apples-to-apples check by evaluating the quantized roundtrip artifact, not only the in-memory training weights.

## Methodology

The work was split into four local stages:

### 1. Phase 0 Sanity Check

The published top-1 script was isolated into this experiment folder and run locally to validate the end-to-end pipeline. Direct local fp16 smoke runs were unstable, so the local proxy setup was standardized on:

- `COMPUTE_DTYPE=fp32`
- `USE_TORCH_COMPILE=0`
- `USE_FUSED_OPTIM=0`
- reduced local learning rates
- gradient clipping for ternary-heavy runs

This produced a stable local baseline and confirmed that proxy data loading, training, export, and post-export evaluation all worked.

### 2. Small 3-Seed Matrix

A small proxy matrix was run across:

- baseline small int5
- full ternary
- mixed ternary
- deep full ternary

This stage was designed to answer one question quickly: which ternary branch deserves larger runs?

### 3. Matched `10L` Comparison

Two matched-shape proxy runs were then compared directly:

- `10L` int5 baseline
- `10L` mixed ternary

This removed model-shape confounders and isolated the compression-vs-quality tradeoff.

### 4. Budget-Filling Mixed Run

Because same-shape mixed ternary saved a large number of bytes, the next run spent some of that savings on a larger mixed model:

- `11` layers
- `d=640`
- `8` query heads
- `4` KV heads
- `MLP_MULT=3`
- `42,630,489` parameters

This is the most important local result so far.

## Results

### Phase 0

- `phase0_baseline_smoke_stable`
  - `val_bpb=4.10415833`
  - quantized artifact `9,229,835` bytes

This run only validated the pipeline. It is not meaningful as a performance target.

### Small 3-Seed Matrix

| Group | Config | Mean val_bpb | Std | Mean quantized bytes | Mean zero frac |
|---|---|---:|---:|---:|---:|
| Baseline small int5 | `int5`, 3L, `d=256` | 4.055352 | 0.001231 | 1,330,341 | n/a |
| Full ternary | `ternary`, 3L, `d=256` | 4.059598 | 0.001478 | 339,876 | 0.566 |
| Mixed ternary | `mixed`, 3L, `d=256` | 4.055295 | 0.001233 | 1,177,475 | 0.655 |
| Deep ternary14 | `ternary`, 14L, `d=256` | 4.088843 | 0.000453 | 1,099,475 | 0.568 |

Interpretation:

- Full ternary bought the largest byte savings, but quality was worse.
- Mixed ternary matched the small int5 baseline within noise.
- Simply increasing depth under full ternary did not help; it hurt quality materially.

That was enough to deprioritize full ternary as the main next branch.

### Matched `10L` Proxy Runs

| Run | Config | Params | Final val_bpb | Total submission bytes |
|---|---|---:|---:|---:|
| `proxy_int5_10l_seed42_20260323` | `int5`, matched shape | 25,517,137 | 3.26066687 | 15,406,268 |
| `proxy_mixed_10l_seed42_20260323` | `mixed`, matched shape | 25,517,137 | 3.26632616 | 8,705,415 |

Interpretation:

- Same-shape mixed ternary lost about `0.00565929` bpb versus int5.
- That same run saved `6,700,853` bytes of total artifact size.

This result was not good enough to claim mixed ternary was better. It did show that same-shape comparison is the wrong end state; the saved bytes have to be spent on a larger model.

### Larger Mixed Runs

| Run | Config | Params | Final val_bpb | Total submission bytes |
|---|---|---:|---:|---:|
| `proxy_mixed_11l640_seed42_20260329` | `mixed`, `11L d=640` | 42,630,489 | 3.26090698 | 13,609,639 |
| `proxy_mixed_11l640_seed1337_20260329` | `mixed`, `11L d=640` | 42,630,489 | 3.26105237 | 13,572,709 |

Derived comparison against matched `10L` int5:

- two-seed mean delta bpb: about `+0.00031`
- bytes saved versus int5: about `1.80 MB`
- parameter ratio: `1.6707x`
- compressed model bytes per parameter:
  - int5: `0.601`
  - larger mixed: about `0.317`

Interpretation:

- These runs improved over the earlier matched-shape mixed run by about `0.00535` to `0.00542` bpb.
- They nearly closed the gap to the int5 baseline while remaining comfortably under the byte cap.
- On the local proxy split, this is the first result that makes the mixed ternary direction look genuinely competitive rather than merely interesting.
- The close agreement between seeds `42` and `1337` is the strongest stability signal seen so far for this branch.

## Compression and Packing Observations

The results match the initial concern that raw bit width is not the whole story. Ternary packing wins on raw representation, but the actual comparison is governed by compressed artifact size after serialization and compression.

Observed local behavior:

- matched `10L` int5 total size: `15.41 MB`
- matched `10L` mixed total size: `8.71 MB`
- larger `11L d=640` mixed total size: `13.61 MB`

The larger mixed run demonstrates the real value proposition: mixed ternary does not need to beat int5 at the same shape. It needs to turn saved bytes into enough extra capacity to recover the quality gap.

## Ternary Sparsity Behavior

Mixed runs consistently stabilized around `0.655` overall zero fraction. That is below the configured `0.80` abort threshold, so runs were not globally collapsing.

However, the worst layer repeatedly saturated at:

- `blocks.9.mlp.proj:1.000` in the matched `10L` mixed run
- `blocks.10.mlp.proj:1.000` in the larger `11L d=640` mixed run

That pattern matters. It suggests the last MLP projection is especially vulnerable under ternary QAT and may be giving up an entire layer-local degree of freedom. The fact that the larger mixed run still remained competitive means this issue is not fatal, but it is the clearest technical weakness in the current mixed design.

## What We Know Now

1. The implementation works end to end on a realistic proxy workflow.
2. Full ternary is not the best near-term path. It saves many bytes, but quality loss is too large in current form.
3. Mixed ternary is the only promising ternary branch.
4. Same-shape mixed ternary is not sufficient, but larger mixed models can almost recover the quality gap.
5. The best local result so far is a larger mixed model that is essentially tied with int5 on proxy quality while still under budget.

## Limits of the Current Evidence

The current evidence is still limited in four important ways:

1. The decisive larger mixed result has only been checked on two seeds so far.
2. The proxy dataset is development-only and may not preserve ranking perfectly against official FineWeb validation.
3. Training time was measured on a local `1x RTX 2060`, not on `8xH100`.
4. The current mixed path still shows per-layer saturation in the last MLP projection.

Because of these limits, this work supports "continue the branch" rather than "declare success."

## Recommended Next Steps

### Immediate

Run the larger mixed configuration on seed `2024`. Seed `1337` now agrees closely with seed `42`, so the immediate question is whether that behavior holds across a third seed.

### If Seed Stability Holds

If the larger mixed configuration remains close to int5 across seeds, test one more controlled scale-up to use more of the remaining budget. The current best run still leaves about `2.39 MB` unused under the `16 MB` cap.

### If Seed Stability Fails

If the larger mixed result regresses materially on other seeds, keep mixed ternary as a useful negative-result study, but stop investing in architecture scaling and instead investigate the saturated final MLP projection.

## Bottom Line

This direction should continue, but only in its mixed form.

The local proxy evidence no longer supports "full ternary versus int5." It supports a narrower thesis: ternary MLP compression may be useful if it is treated as a budget reallocation tool that funds a larger model while keeping attention at higher precision.
