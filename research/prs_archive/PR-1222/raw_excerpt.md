# PR 1222 — Prime MLP Test-Time Training: Naive vs E2E (FOMAML)

**Author:** (from track_non_record_16mb, see git log)
**Claimed BPB:** Naive TTT eval-only best 1.2906 (baseline 1.3696, delta -0.079); FOMAML+TTT best 1.2720 (delta -0.097); on a weak 40-shard baseline model.
**Seeds:** not stated
**Hardware:** 1x L40S (40 shards, MLP 3.5x, 10K steps)

## Files retrieved
- `records__track_non_record_16mb__2026-04-01_TTT_E2E_FOMAML__README.md`
- `records__track_non_record_16mb__2026-04-01_TTT_E2E_FOMAML__train_e2e_proper.py`
- `records__track_non_record_16mb__2026-04-01_TTT_E2E_FOMAML__train_ttt_e2e.py`

## Claimed changes (from README, verbatim)

> Two studies on test-time training with prime MLP adapters. Naive TTT gives -0.079 BPB for free (eval-only). E2E FOMAML gives -0.097 total but costs 44% of the training budget.

Motivation: All 25 prior naive TTT attempts failed because they perturbed GPTQ'd int5/int6 weights. Prime MLPs are separate bf16 parameters — they don't touch GPTQ'd weights.

Architecture: Rank-256 prime MLPs on all 11 blocks, running before the main MLP. Down projection zero-init. Score-first eval is legal.

Head-to-head:
- Naive TTT (eval-only): 1.3696 -> 1.2906, delta -0.079, 0 training cost.
- FOMAML + TTT: 1.3696 -> 1.2720, delta -0.097, 3000 steps (~44% budget).

Key findings:
- Layer count >> rank. All 11 layers (-0.045) beats rank=512 on 3 layers (-0.006).
- Higher LR better up to 1.0.
- Joint FOMAML improves model -0.260 BPB from FOMAML baseline even without TTT.
- TTT nearly redundant after FOMAML — only -0.001 additional.

Note: Baseline BPB here is 1.3696 (weak model), not competitive frontier. Next steps list includes 8xH100 validation on PR 1105 model (1.1125 BPB) — not yet done.
