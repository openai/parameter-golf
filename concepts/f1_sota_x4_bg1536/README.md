# F1 SOTA + 2 Safe Speed Knobs

This is an isolated copy for a speed-safe test.

## Baseline Provenance

- Source commit: `303192e9ac65fa1673de647b02d1bb7365c37198`
- Source file: repository root `train_gpt.py`
- Intent: start from the same SOTA baseline referenced for PR #587, not from the modified F1 experimental branch.

## Only Two Additions

Applied as runtime env overrides in `run.sh` (no code changes in `train_gpt.py`):

1. `XSA_LAST_N=4`
2. `BIGRAM_VOCAB_SIZE=1536`

Everything else remains baseline behavior.

## Run

```bash
SEED=1337 bash concepts/f1_sota_x4_bg1536/run.sh
```
