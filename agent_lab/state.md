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

## Open Questions

- Is `9L / MLP2 / 131072` already the right frontier shape, or can another structural trim beat it?
- Do the remaining tranche-C runs teach anything useful beyond the new winner, especially about whether one less layer is a cleaner size cut than one less MLP notch?
- After tranche C closes, does the new winner want more steps, a small LR retune, or another minor size-capacity trade?

## Next Planned Runs

- Tranche C asks one question:
  Can width-oriented near-misses be made valid by trimming the right structure, not just shrinking everything?
- Completed:
- `C1-E1`: `9L / MLP3 / DIM480 / 131072 / kv2` -> valid but weak
- `C1-E3`: `9L / MLP2 / 512 / 131072 / kv2` -> new best valid frontier
- `C1-E2`: `9L / MLP3 / DIM448 / 131072 / kv2` -> valid and much stronger than `DIM480`, but still behind `9L / MLP2`
- Remaining:
- `C1-E4`: `8L / MLP3 / 512 / 131072 / kv2`
- `C1-E5`: `8L / MLP3 / DIM480 / 131072 / kv2`
- Tranche D will be rewritten after tranche C finishes so the follow-up runs target the actual survivors instead of the stale draft.

## Go Deeper

- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Architecture audit: [`architecture_review.md`](./architecture_review.md)
- Experiment dashboard: [`plots/experiments.html`](./plots/experiments.html)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Latest narrative log: [`docs/build-logs/2026-03-29-agent-lab.md`](../docs/build-logs/2026-03-29-agent-lab.md)
