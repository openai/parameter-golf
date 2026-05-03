# V14: PR #1735 + TTT Weights EMA

**Base:** PR #1735 (AjAnubolu, 1.0429 BPB) — SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + 8-GPU Parallel Pre-Quant AdamW TTT

**Innovation:** Add EMA averaging to the 21-epoch pre-quant TTT phase. Instead of using the final epoch's weights, use an exponentially-weighted moving average across all epochs.

## Why This Should Help

AjAnubolu's TTT runs 21 epochs of AdamW with cosine LR (5e-4 -> 5e-5). At convergence, weights oscillate around a local optimum. Using only the LAST epoch's weights captures the noise. EMA averaging:

1. Smooths out late-epoch oscillation
2. Effectively averages multiple "good" local optima
3. Costs <1 second of compute and 0 bytes in artifact
4. Standard ML technique (used in DeepMind, OpenAI, Meta papers)

## Compliance

- **Inherits PR #1735's compliance status** (pre-quant TTT framework)
- **No additional risk**: EMA is a fixed averaging procedure, not val-loss-based selection
- **No new training**: just averages weights from existing 21 epochs

## Implementation

Two new env vars:

```bash
TTT_EMA_ENABLED=1   # default: 1 (on)
TTT_EMA_DECAY=0.7   # default: 0.7 (effective last-5-epochs window)
```

EMA logic (added to `pre_quant_adamw_ttt`):

```python
# Init: clone trainable params
ttt_ema_state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

# Each epoch: EMA update after all_reduce sync
for n, p in model.named_parameters():
    if n in ttt_ema_state:
        ttt_ema_state[n].mul_(0.7).add_(p.data, alpha=0.3)

# After all epochs: replace model with EMA
for n, p in model.named_parameters():
    if n in ttt_ema_state:
        p.data.copy_(ttt_ema_state[n])
```

## Usage on RunPod

```bash
# Clone this branch
cd /workspace
git clone -b v14-pr1735-ttt-ema https://github.com/alertcat/parameter-golf.git
cd parameter-golf

# Install deps (same as PR #1735)
pip install sentencepiece brotli zstandard
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Download SP8192 data
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# Train + eval (TTT EMA enabled by default)
cd records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/
SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Decision Points During Run

Watch for these log lines (TTT phase, last ~6 minutes of run):

```
prequant_ttt:start epochs=21 lr=0.0005 ...
ttt_ema:initialized decay=0.7 params=NN
prequant_ttt:epoch 1/21 val_bpb=1.06X ...
prequant_ttt:epoch 21/21 val_bpb=1.034X ...   <- last epoch (baseline)
ttt_ema:loaded final EMA weights into model
ttt_ema:final val_bpb=1.0XX                   <- our metric (should be lower)
```

If `ttt_ema:final val_bpb` is **lower** than `prequant_ttt:epoch 21/21 val_bpb` -> EMA helped.
Then GPTQ quantizes the EMA weights, runs sliding eval -> final number.

## Expected Results

| Metric | PR #1735 (base) | V14 (this PR) | Delta |
|--------|----------------:|--------------:|------:|
| Pre-quant val_bpb | 1.034 | ~1.032 | -0.002 |
| Final sliding val_bpb | 1.0429 | ~1.040-1.042 | -0.001 to -0.003 |
| Artifact size | 15,991,294 | ~15,992,000 | ~+1KB (negligible) |

3-seed mean target: **1.040 BPB**

## Hyperparameter Tuning (if scout shows promise)

Try in this order:
1. `TTT_EMA_DECAY=0.5` (faster decay, last-3-epochs)
2. `TTT_EMA_DECAY=0.85` (slower, last-7-epochs)
3. `TTT_EMA_DECAY=0.95` (very slow, broad average)

## File Changes

- `records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/train_gpt.py`: +60 lines (4 patch sites in `pre_quant_adamw_ttt`)
- `patch_v14_ttt_ema.py`: standalone patch script (regenerable)
- `V14_README.md`: this file

Net diff: ~+1500 bytes
