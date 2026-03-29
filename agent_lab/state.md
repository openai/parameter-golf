# Agent Lab State

This is the first-read dashboard for autonomous research. Read this file for the high-level picture, then drill down into the linked tranche, idea bank, experiment ledger, and build log.

## Current Best Valid

- Experiment: [`AL-20260329-012`](./experiments.tsv)
- Commit: `d99bcaa`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3838`
- Winning branch shape: `9` layers, `MLP_MULT=2`, `MODEL_DIM=512`, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=131072`
- Total artifact size: `14,733,304` bytes int8+zlib

## Best Invalid Near-Miss

- Experiment: [`AL-20260329-010`](./experiments.tsv)
- Commit: `fe477f9`
- Primary metric: `final_int8_ttt_lora`
- Best invalid `val_bpb`: `1.3899`
- Shape: `9` layers, `MLP_MULT=3`, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=131072`
- Why invalid: `17,680,105` bytes int8+zlib exceeds the `16,000,000` byte cap

## Active Tranche

- Tranche: [`T-20260329-C`](./tranches.md#t-20260329-c-width-winner-size-recovery)
- Goal: finish the size-recovery sweep around the width-biased near-miss and identify the best valid survivor for tranche D
- Status: active

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

## Open Questions

- Is `9L / MLP2 / 131072` already the right frontier shape, or can another structural trim beat it?
- Does the winner `9L / MLP2 / 131072` still want more steps, a small LR retune, or both together?
- Is the backup `8L / MLP3 / DIM480 / 131072` merely smaller, or can optimization make it genuinely competitive?
- Which of those two survivors deserves to anchor the next architecture tranche after optimization is explored?

## Next Planned Runs

- Tranche C is complete.
- Best survivor:
- `9L / MLP2 / 512 / 131072 / kv2` -> `1.3838`, valid, current frontier
- Best fallback:
- `8L / MLP3 / DIM480 / 131072 / kv2` -> `1.3906`, valid, smaller but clearly weaker
- Tranche D now asks:
  can optimization improve the two actual survivors, especially the new `9L / MLP2` winner?
- Planned tranche-D runs:
- `D1-E1`: `9L / MLP2 / 512 / 98304 / kv2`
- `D1-E2`: `9L / MLP2 / 512 / 131072 / kv2 / MATRIX_LR=0.065`
- `D1-E3`: `9L / MLP2 / 512 / 98304 / kv2 / MATRIX_LR=0.065`
- `D1-E4`: `8L / MLP3 / DIM480 / 98304 / kv2`
- `D1-E5`: `8L / MLP3 / DIM480 / 98304 / kv2 / MATRIX_LR=0.065`

## Go Deeper

- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Architecture audit: [`architecture_review.md`](./architecture_review.md)
- Experiment dashboard: [`plots/experiments.html`](./plots/experiments.html)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Latest narrative log: [`docs/build-logs/2026-03-29-agent-lab.md`](../docs/build-logs/2026-03-29-agent-lab.md)
