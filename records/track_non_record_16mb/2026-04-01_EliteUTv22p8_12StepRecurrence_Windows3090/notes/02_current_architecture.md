## 02 — Current Architecture

Code references:

- `model.py`
- `triton_mlp.py`

## High-level model shape

The current model is a **causal decoder LM** with a **tied recurrent block**.

### Core flow (`GPT.forward_logits`)

1. Token embedding: `tok_emb(input_ids)`
2. Optional pre-block feature paths:
   - smeargate
   - bigram hash
3. Recurrent loop over `num_steps`:
   - add step embedding
   - optional level signal
   - run shared `Block`
4. Final norm + output projection (tied or untied head)
5. Logit softcap (`tanh`-based)

## Why it is not “plain UT” anymore

The tied-depth idea remains, but each step is strongly modulated by per-step pathways:

- Step embeddings (`step_embeddings[i]`)
- Per-step LoRA adapters in Q/V/MLP (`RelaxedLinear`)
- Per-step value bias (`v_step_bias`)
- Optional per-step low-rank level signal

This means recurrence is no longer homogeneous even though main dense weights are tied.

## Block internals

`Block` contains:

- Pre-norm attention branch (`attn_norm -> CausalSelfAttention`)
- Pre-norm MLP branch (`mlp_norm -> MLP`)
- Residual updates with learned `attn_scale`/`mlp_scale`
- Dropout and subtle stochastic depth behavior in training path

## Attention details

`CausalSelfAttention` includes:

- RoPE (`Rotary`)
- grouped KV heads (`num_kv_heads`)
- learned per-head `q_gain`
- optional per-step V bias and optional Q/V LoRA adapters

## MLP details

- Main path uses Triton fused op (`fused_relu2`) from `triton_mlp.py`
- Supports step-conditioned LoRA contributions
- Projection layer initialized for stable residual startup
