# Partial RoPE (25% of Head Dimensions)

## Score: val_bpb = TBD

## Hypothesis

Applying RoPE to only 16/64 head dimensions allows the remaining 48 dimensions to learn position-independent semantic similarity. Community data shows ~0.005 BPB improvement. Zero additional parameters, zero compute overhead.

## Change from exp27

Single architectural change in `CausalSelfAttention`:
- `rope_frac=0.25`: RoPE applied to first 16 dims, remaining 48 pass through unchanged
- `Rotary` module initialized with `dim=16` instead of `dim=64`

## Architecture

| Parameter | Value |
|-----------|-------|
| num_layers | 11 (10 unique) |
| model_dim | 512 |
| num_heads | 8, num_kv_heads | 4 |
| head_dim | 64 (16 RoPE + 48 pass-through) |
| mlp_mult | 3.0 (hidden=1536) |
| mlp_activation | LeakyReLU(0.5)² |
| rope_frac | 0.25 |

## Expected Impact

~0.005 BPB improvement over exp27 baseline (1.3345 → ~1.330)

## Results

TBD — awaiting A100 run.
