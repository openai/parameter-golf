# Spec 000 — SOTA Replication

**Slug:** `sota-replication`
**Created:** 2026-04-19
**Links to idea:** N/A (this is the baseline, not an idea)

## Hypothesis
`train_gpt_sota.py` as-is on 8×H100 NA-1 with seed 42 reproduces the official SOTA submission at **1.0810 ± 0.002 bpb**. If it doesn't replicate, every subsequent Δ is measured against a moving target and further experiments are pointless.

## Baseline
Official leaderboard entry `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` at 1.0810 bpb. Reference: `records/track_10min_16mb/2026-04-09_*/README.md`.

## Expected Δ
Δ = 0 (this is the replication target, not an improvement). Success = landing within ±0.002 of 1.0810.

## Accept criteria
- Single run, 8×H100, seed 42: `val_bpb ∈ [1.079, 1.083]`.
- No NaN, no divergence, wall time within 20% of SOTA submission's ~588s.

## Config diff
One override: **`BIGRAM_VOCAB_SIZE=0`** to disable the BigramHash embedding that's present in our `train_gpt_sota.py` (commit `d529fe8`) but NOT in the leaderboard SOTA submission. Defaults otherwise enable BigramHash at 3072×112 (see `train_gpt_sota.py:97-98, 432-474`), which would mean running a different model than the 2026-04-09 reference. Setting `BIGRAM_VOCAB_SIZE=0` gates `self.bigram` to `None` (L474) and matches the leaderboard.

Otherwise: shipped defaults — SP8192 tokenizer, 11L / 512d / MLP4×, 3-layer depth recurrence, parallel residuals from layer 7, LeakyReLU², MuonEq-R, GPTQ INT6, SDClip, EMA, TTT enabled, sliding window, brotli-11.

## Code changes
- Branch: `research`
- Commit: `01e6fcf` (HEAD at spec freeze)
- Diff: none

## Hardware ladder
- [ ] 2×H100 mini — **skip**, Exp 24 already validated the SOTA code path on 2×H100
- [ ] 8×H100 official — seed 42 (only rung)

The *new* surface area vs. Exps 23–24 is checkpoint I/O (rank-0 `torch.save` × 9). Covered by the first-minute preflight under Stop-early, not a dedicated smoke.

## Seed plan
Single seed (42). Full 3-seed submission is a separate future spec once we have Δ improvements to submit.

## Inputs
- Data: `/workspace/data/datasets/fineweb10B_sp8192/` (verify exact path during interview)
- Tokenizer: `/workspace/data/tokenizers/fineweb_8192_bpe.model`
- Hotstart: none
- Base repo commit: `01e6fcf` on `research`

## Checkpoints to emit
Phase-boundary set, emitted automatically when `CKPT_DIR` is set. Purpose: reusable hotstart seeds for downstream experiments — a run can pick up at any phase boundary and only retrain the tail.

| Filename pattern | When | `train_gpt_sota.py` | Downstream use |
|---|---|---|---|
| `ckpt_event_step{N}.pt` (momentum warmup end) | `h.muon_momentum_warmup_steps` | L1299 | LR / Muon-momentum experiments |
| `ckpt_pre_recurrence_step{N}.pt` | frac ≥ 0.35 | L1349 | recurrence-policy / layer-index swaps |
| `ckpt_warmdown_start_step{N}.pt` | `lr_mul(frac) < 1.0` | L1346 | warmdown-shape / TTT tweaks |
| `ckpt_final_pre_ema_step{N}.pt` | end of training loop | L1380 | EMA-decay / post-hoc EMA |
| `ckpt_final_post_ema_step{N}.pt` | after EMA applied | L1385 | quant-only experiments |
| `ckpt_event_step{N}.pt` × 4 | `CKPT_STEPS=455,1137,2275,3412` (~10/25/50/75% of 4550) | L1359–1360 | partial-train probes |

- **State per ckpt:** model + optimizer states (Muon + Adam) + EMA state (`save_checkpoint`, L1312–1317).
- **Size per ckpt:** ~1 GB. **Total:** ~9 GB.
- **Retention:** keep all through the record-track push (through 2026-04-30). Policy revisited after.
- **Destination:** `/workspace/runs/000-sota-replication/checkpoints/` (directory, not a single file).

## Stop-early criteria
- NaN in train loss at any step → kill, mark failed.
- val_bpb at midpoint eval > 1.5 → kill, probably broken.
- Step time > 2× expected → kill, investigate.
- **Ckpt-I/O preflight fail:** if the `Checkpoints will be saved at steps: [...]` log line doesn't appear on rank 0 within the first minute, or the momentum-warmup ckpt doesn't land on disk within ~60s of training start, kill before step 500 — `CKPT_DIR` is not plumbed through.

## Cost estimate
- 8×H100 NA-1, ~10 min **training** / ~12 min **total wall** (compile + evals + EMA/quant/compress + ckpt writes): **~$3.50**.

## Extra artifacts
The 9-file phase-boundary checkpoint set IS the artifact beyond defaults. No separate extras.

## Open questions for interview
- Confirm current `research` HEAD (`01e6fcf`) is the intended SOTA code (not a stale branch).
- Confirm SP8192 data exists on the NA-1 volume, not just SP1024. If missing, need a data-prep spec first.
- Confirm 8×H100 NA-1 availability before provisioning.
- Confirm ≥ 15 GB free on NA-1 scratch for the checkpoint set.
- Confirm `CKPT_DIR=/workspace/runs/000-sota-replication/checkpoints/` and `CKPT_STEPS=455,1137,2275,3412` are set in the launch env before `torchrun`.
- Confirm `BIGRAM_VOCAB_SIZE=0` is set in the launch env (overrides the default 3072 that's baked into our code but absent from the leaderboard SOTA).
