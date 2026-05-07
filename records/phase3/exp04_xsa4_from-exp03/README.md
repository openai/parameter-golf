# Exclusive Self-Attention (XSA) on Last 4 Layers

## Score: val_bpb = TBD

## Hypothesis

XSA removes the self-value bias from attention output by subtracting `v_i / seq_len` from each position. This forces the model to rely on information from other tokens rather than copying its own value. Applied to last 4 of 10 unique layers (not all). Zero additional parameters, slight compute overhead.

Community shows XSA + EMA is the "prerequisite stack" for all frontier techniques. EMA without XSA loses 0.003 BPB; EMA with XSA gains 0.003 BPB.

## Changes from exp03

- `CausalSelfAttention` gains `use_xsa` flag
- When enabled, subtracts `v_expanded / seqlen` after SDPA (removes self-value contribution)
- `Block` passes `use_xsa` to attention
- `GPT` enables XSA on last `xsa_last_n=4` physical blocks

## Architecture

Inherits from exp03 (Partial RoPE + LN Scale + EMA) + XSA on last 4 layers.

## Expected Impact

~0.01-0.02 BPB improvement over exp03. Also synergizes with EMA.

## Results

TBD — awaiting A100 run.
