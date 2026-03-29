# Agent Lab State

This is the first-read dashboard for autonomous research. Read this file for the high-level picture, then drill down into the linked tranche, idea bank, experiment ledger, and build log.

## Current Best

- Experiment: [`AL-20260329-004`](./experiments.tsv)
- Commit: `41e9478`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3913`
- Winning branch shape: `10` layers, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=131072`
- Total artifact size: `15,824,353` bytes int8+zlib

## Active Tranche

- Tranche: [`T-20260329-B`](./tranches.md#t-20260329-b---architecture-necessity-audit)
- Goal: start the architecture necessity audit, beginning with a focused MLP-width-versus-depth tranche
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

## Open Questions

- Is the tiny `131072` edge over `196608` real or just noise?
- Can we preserve the `10`-layer gain while reclaiming artifact headroom?
- Can the `9L / MLP3` near-miss turn into a win if we recover more steps, or is width still fundamentally behind depth here?

## Next Planned Runs

- `B1-E5`: test `9L / MLP3 / 131072 / kv2` to ask whether width also needs step recovery.

## Go Deeper

- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Architecture audit: [`architecture_review.md`](./architecture_review.md)
- Experiment dashboard: [`plots/experiments.html`](./plots/experiments.html)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Latest narrative log: [`docs/build-logs/2026-03-29-agent-lab.md`](../docs/build-logs/2026-03-29-agent-lab.md)
