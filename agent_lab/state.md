# Agent Lab State

This is the first-read dashboard for autonomous research. Read this file for the high-level picture, then drill down into the linked tranche, idea bank, experiment ledger, and build log.

## Current Best Valid

- Experiment: [`AL-20260329-021`](./experiments.tsv)
- Commit: `2e78e06`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3709`
- Winning branch shape: `9` layers, `MLP_MULT=2`, `MODEL_DIM=512`, `NUM_HEADS=4`, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=98304`
- Total artifact size: `15,329,587` bytes int8+zlib

## Best Invalid Near-Miss

- Experiment: [`AL-20260329-010`](./experiments.tsv)
- Commit: `fe477f9`
- Primary metric: `final_int8_ttt_lora`
- Best invalid `val_bpb`: `1.3899`
- Shape: `9` layers, `MLP_MULT=3`, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=131072`
- Why invalid: `17,680,105` bytes int8+zlib exceeds the `16,000,000` byte cap

## Active Tranche

- Tranche: [`T-20260329-E`](./tranches.md#t-20260329-e-attention-geometry-audit)
- Goal: test whether the new `9L / MLP2 / 98304` frontier is now limited more by attention geometry than by the already-mapped MLP or optimizer knobs
- Status: planned next

## Working Beliefs

- The mar29 system runtime is stronger than the earlier mar28 local anchor, so new baselines must be compared against the refreshed local frontier, not the older numbers.
- `10` layers are useful only when they are not step-starved.
- Smaller batches recovered enough optimizer steps to turn the `10`-layer branch from a loser into the current winner.
- The frontier is flattening: `196608` and `131072` batch settings are very close.
- Cheaper attention via `NUM_KV_HEADS=1` was not a better trade than keeping `NUM_KV_HEADS=2`.
- At fixed `10`-layer depth, shrinking the MLP to `MLP_MULT=1` bought steps and artifact headroom, but not enough quality to beat the `MLP_MULT=2` anchor.
- At fixed `10`-layer depth, widening to `MLP_MULT=3` was clearly bad: fewer steps, worse `val_bpb`, and a 17.43 MB artifact that breaks the cap.
- Reallocating width into one extra layer via `11L / MLP1` did not rescue the idea; it stayed within size and got good step count, but still lost on quality.
- The first promising width branch is `9L / MLP3`: it improved a lot over `10L / MLP3`, but it still trails the anchor and is slightly too large at 16.18 MB.
- Width clearly benefits from step recovery: `9L / MLP3 / 131072` became the best raw scorer at `1.3899`, but the artifact grew even further over the cap.
- A mild global dim trim to `DIM480` is enough to make the width winner valid on size, but not enough to keep it competitive on score.
- The cleaner byte cut is not a global dim trim. `9L / MLP2 / 131072` kept the high-step regime, stayed well under the size cap, and moved the valid frontier to `1.3838`.
- If a global dim trim is necessary, `DIM448` is the meaningful point, not `DIM480`: it recovered most of the lost score and stayed very small, but it still did not beat the `9L / MLP2` survivor.
- Dropping one layer is not the cleaner size move here. `8L / MLP3 / 131072` kept decent quality, but it was still over the cap, so the layer cut saved fewer useful bytes than the MLP-notch cut.
- The best fallback after the winner is the combined-light-cuts line, `8L / MLP3 / DIM480 / 131072`. It is valid and stronger than the other fallback trims, but it still sits well behind `9L / MLP2 / 131072`.
- The winner was still substantially step-limited. Dropping batch to `98304` on `9L / MLP2` was a major gain, not a marginal one, and it still stayed inside the artifact cap.
- A simple LR bump is not interchangeable with extra steps. `MATRIX_LR=0.065` at `131072` batch fell back to the old frontier band instead of following the `98304` win.
- The combo test also failed to beat the step-only win. `98304 + MATRIX_LR=0.065` stayed strong, but it still lost clearly to `98304` with the default LR.
- The fallback line is also step-limited. `8L / MLP3 / DIM480 / 98304` improved a lot over its `131072` version, but it still remains secondary to the `9L / MLP2 / 98304` frontier.
- The LR bump helps neither survivor. It regressed both the main line and the fallback line, so the current best-tested setting keeps the default `MATRIX_LR=0.06`.
- The next good question is no longer optimization. The best line is now stable enough that another worthwhile tranche should attack a different component family.
- Attention is the cleanest next component family because it has several env-exposed geometry knobs: query-head count, KV sharing, and QK sharpness at initialization.
- The first attention-geometry probe already moved the frontier. `q4/kv2` beat `q8/kv2`, which is a strong hint that the current model wants fewer, wider query heads.
- The opposite head-direction failed hard. `q16/kv2` regressed badly, which means the E1 result is directional evidence, not a generic “head changes help” result.

## Open Questions

- Is the new frontier now limited more by attention geometry than by feedforward capacity or optimizer choice?
- Does the model clearly want fewer wider query heads, or is `q4` just one side of a broader head-geometry optimum?
- Is the current `QK_GAIN_INIT=1.5` making attention too sharp or too flat for the `98304`-token regime?

## Next Planned Runs

- Tranche C taught us:
- the right byte-saving move was `MLP3 -> MLP2`, not a dim trim or a layer cut
- Tranche D taught us:
- the winning `9L / MLP2` line was still step-limited at `131072`
- more steps helped both survivors
- the `MATRIX_LR=0.065` bump helped neither
- Tranche E now asks:
  is the current `9L / MLP2 / 98304 / q8-kv2` frontier leaving quality on the table because its attention geometry is mis-set?
- Planned tranche-E runs:
- `E1`: `NUM_HEADS=4, NUM_KV_HEADS=2` -> complete, new frontier at `1.3709`
- `E2`: `NUM_HEADS=16, NUM_KV_HEADS=2` -> complete, clear regression
- `E3`: `NUM_HEADS=8, NUM_KV_HEADS=4`
- `E4`: `QK_GAIN_INIT=1.0`
- `E5`: `QK_GAIN_INIT=2.0`

## Go Deeper

- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Architecture audit: [`architecture_review.md`](./architecture_review.md)
- Experiment dashboard: [`plots/experiments.html`](./plots/experiments.html)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Latest narrative log: [`docs/build-logs/2026-03-29-agent-lab.md`](../docs/build-logs/2026-03-29-agent-lab.md)
