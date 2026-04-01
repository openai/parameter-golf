# Non-record: Universal Transformer + Adaptive Density

**val_bpb: 1.4390** | 1×RTX 5090, 600s + TTT

Single shared transformer block looped 12× with per-iteration params (attn_scale, mlp_scale, resid_mix, iteration_embed). 50% sparse→dense curriculum. 4.56M params compressed to 2.87MB.

Implements OpenAI's requested 'Universal transformer' direction. Confirms depth recurrence findings from PR #363: shared-weight architectures trade parameter efficiency for BPB quality at this scale.
