# EMA (Exponential Moving Average) Replacing SWA

## Score: val_bpb = TBD

## Hypothesis

EMA with decay=0.997 outperforms SWA by ~0.003 BPB (community verified, 3-seed). EMA updates every step (smoother averaging) vs SWA's periodic snapshots. Critical prerequisite for XSA (exp04).

## Changes from exp02

- Removed SWA hyperparameters (`swa_enabled`, `swa_start_frac`, `swa_every`)
- Added `ema_enabled=True`, `ema_decay=0.997`
- EMA shadow weights updated every training step: `ema = decay * ema + (1-decay) * weights`
- EMA weights loaded before final eval (replaces SWA averaging)

## Architecture

Inherits from exp02 (Partial RoPE + LN Scale) + EMA replacing SWA.

## Expected Impact

~0.003 BPB improvement over exp02. Also unlocks synergy with XSA.

## Results

TBD — awaiting A100 run.
