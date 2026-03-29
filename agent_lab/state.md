# Agent Lab State

This is the first-read dashboard for autonomous research. Read this file for the high-level picture, then drill down into the linked tranche, idea bank, experiment ledger, and build log.

## Current Best Valid

- Experiment: [`AL-20260329-004`](./experiments.tsv)
- Commit: `41e9478`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3913`
- Winning branch shape: `10` layers, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=131072`
- Total artifact size: `15,824,353` bytes int8+zlib

## Best Raw Score But Invalid

- Experiment: [`AL-20260329-010`](./experiments.tsv)
- Commit: `fe477f9`
- Primary metric: `final_int8_ttt_lora`
- Best raw `val_bpb`: `1.3899`
- Shape: `9` layers, `MLP_MULT=3`, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=131072`
- Why invalid: `17,680,105` bytes int8+zlib exceeds the `16,000,000` byte cap

## Active Tranche

- Tranche: [`T-20260329-C`](./tranches.md#t-20260329-c-width-winner-size-recovery)
- Goal: recover bytes around the raw width winner and try to turn it into a challenge-valid frontier
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

## Open Questions

- Can the `9L / MLP3 / 131072` winner be pulled back under the size cap without losing its score?
- Is there a cleaner width-oriented shape than `9L / MLP3` that keeps most of the gain without the compression failure?
- If size-recovered candidates lose a bit of score, can optimization recover it?

## Next Planned Runs

- Tranche C asks one question:
  Can the raw width winner be made valid by trimming bytes intelligently?
- `C1-E1`: `9L / MLP3 / DIM480 / 131072 / kv2`
- `C1-E2`: `9L / MLP3 / DIM448 / 131072 / kv2`
- `C1-E3`: `9L / MLP2 / 512 / 131072 / kv2`
- `C1-E4`: `8L / MLP3 / 512 / 131072 / kv2`
- `C1-E5`: `8L / MLP3 / DIM480 / 131072 / kv2`
- Tranche D asks the follow-up question:
  If the size-recovered candidates are valid but slightly weaker, can optimization and extra steps recover the loss?
- `D1-E1`: `9L / MLP3 / DIM480 / 98304 / kv2`
- `D1-E2`: `9L / MLP3 / DIM480 / 131072 / kv2 / MATRIX_LR=0.065`
- `D1-E3`: `9L / MLP2 / 512 / 98304 / kv2`
- `D1-E4`: `8L / MLP3 / 512 / 98304 / kv2`
- `D1-E5`: `8L / MLP3 / DIM480 / 98304 / kv2`

## Go Deeper

- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Architecture audit: [`architecture_review.md`](./architecture_review.md)
- Experiment dashboard: [`plots/experiments.html`](./plots/experiments.html)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Latest narrative log: [`docs/build-logs/2026-03-29-agent-lab.md`](../docs/build-logs/2026-03-29-agent-lab.md)
