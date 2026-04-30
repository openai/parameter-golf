# Non-record: Text Diffusion (MDLM) — Masked Discrete Diffusion

**val_bpb: 3.3801** | 1×RTX 5090, 180s | Signs of life

First diffusion-based submission to Parameter Golf. Hybrid AR+diffusion training: 30% AR loss + 70% masked diffusion loss. Bidirectional attention for masked prediction, causal for AR eval. Cosine masking schedule. Both losses decrease during training.

Implements OpenAI's requested 'Text diffusion' direction.
