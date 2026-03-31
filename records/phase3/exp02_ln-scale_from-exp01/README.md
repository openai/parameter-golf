# LN Scale (1/√(layer+1)) Damping

## Score: val_bpb = TBD

## Hypothesis

Post-RMSNorm output scaling by `1/√(layer_idx+1)` damps deeper layer contributions, preventing later layers from overwriting early representations. Community data shows ~0.003 BPB improvement. Zero additional parameters.

## Changes from exp01

- `Block.__init__` now takes `layer_idx`, computes `self.ln_scale = 1/√(layer_idx+1)`
- `Block.forward` multiplies both `attn_scale` and `mlp_scale` residuals by `ln_scale`

## Architecture

Inherits from exp01 (Partial RoPE 25%) + adds LN Scale damping.

## Expected Impact

~0.003 BPB improvement over exp01.

## Results

TBD — awaiting A100 run.
