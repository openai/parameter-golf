# Candidate Record: PR549 Base + XSA-all + BigramHash3072 + Budget-Aware Export

This directory is a **draft record candidate** built from the merged PR #549 stack, with a narrower goal than the recent high-variance evaluator-heavy PRs:

- stay inside the already-accepted `train_gpt.py` / score-first-TTT framework
- improve the neural stack with low-drama changes that have credible precedent
- leave room for a small legal TTT sweep instead of betting on n-gram caches or disputed eval paths

## Intended Changes

Relative to merged PR #549, this candidate makes four targeted adjustments:

1. **XSA on all 11 layers**
   - default `XSA_LAST_N=11`
   - motivated by PR #634 / PR #728 style results

2. **Bigger BigramHash under the same artifact budget**
   - default `BIGRAM_VOCAB_SIZE=3072`
   - default `BIGRAM_DIM=112`
   - intended to reduce hash collisions without exploding bytes

3. **Budget-aware export pruning**
   - new `TARGET_MB` knob, default `15.90`
   - if the quantized artifact is too large, export prunes the least important `±1` int6 values first
   - this is designed to make wider bigram settings practical instead of failing the size cap outright

4. **TTT optimizer switch for legal score-first sweeps**
   - new `TTT_OPTIMIZER` in `{sgd, adamw}`
   - keeps the merged PR #549 score-first chunked evaluation path
   - this is only an optimizer swap inside the accepted legal protocol, not a pre-adapt/full-val TTT path

## Current Status

- `train_gpt.py` has been updated
- old copied logs were intentionally removed
- `submission.json` is placeholder-only until a fresh run is completed
- no new result is claimed in this directory yet

## Recommended Runs

### 2xH100 cost-efficient tuning run

To approximate the official 8xH100 / 10-minute training budget on a cheaper 2xH100 box, keep the same
global batch and extend the wallclock cap to roughly 40 minutes:

```bash
bash run_2xh100_budget.sh
```

This keeps the effective batch shape aligned with the official recipe because this code fixes
`grad_accum_steps = 8 / WORLD_SIZE`. With `WORLD_SIZE=2`, each optimizer step uses four micro-steps and
roughly matches the total token budget of the 8-GPU run, just at lower hardware parallelism.

### 8xH100 official-style run

For a record-attempt run, use the official 10-minute training cap on 8 GPUs:

```bash
RUN_ID=pr549_xsa11_bigram3072_smoke \
BIGRAM_VOCAB_SIZE=3072 \
BIGRAM_DIM=112 \
XSA_LAST_N=11 \
TARGET_MB=15.90 \
TTT_ENABLED=1 \
TTT_OPTIMIZER=sgd \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_FREEZE_BLOCKS=0 \
VAL_LOSS_EVERY=999999 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Then run the smallest legal TTT sweep first:

```bash
TTT_OPTIMIZER=adamw
TTT_LR=0.0001
TTT_WEIGHT_DECAY=0.01
TTT_EPOCHS=3
TTT_FREEZE_BLOCKS=8
TTT_CHUNK_TOKENS=131072
```

## FlashAttention

This candidate auto-detects whether FlashAttention 3 should be used:

- on H100 (`sm90`), `FLASH_ATTN3_MODE=auto` enables FA3 if the wheel is installed
- on unsupported GPUs, the script automatically falls back to PyTorch SDPA

For the current `cp311 + torch2.10 + cu128` environment, the working install command is:

```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2100
```

## Why This Candidate Exists

The official merged record is still PR #549 at `1.1194`. To be accepted as a new record, a new run needs to beat that by enough margin to clear the competition significance rule, which effectively means aiming closer to `1.112x` than `1.118x`.

This candidate is trying to get there without leaning on:

- n-gram cache legality debates
- shared-table eval systems
- pre-adapt TTT that no longer matches the merged rule interpretation
- val-calibration claims that may be scrutinized

## Files

- `train_gpt.py` — updated candidate training/eval script
- `README.md` — this note
- `submission.json` — placeholder metadata to overwrite after a real run
