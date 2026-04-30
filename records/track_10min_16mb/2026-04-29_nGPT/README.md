# Submission: nGPT implementation

Implemented a normalized GPT where all hidden states and weight matrices are constrained to the unit hypersphere ([paper](https://arxiv.org/pdf/2410.01131)). Every weight matrix has unit-norm rows (re-projected after each optimizer step), and every hidden state is L2-normalized.

## Results

Key metrics (from `train.log`):
- Timed training stopped at `8959/20000` steps due to the wallclock cap.
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_loss:2.07902809 val_bpb:1.23131754`
- Train time: `600080ms` (`step_avg:66.98ms`)
- Peak memory: `15382 MiB allocated`, `16268 MiB reserved`
- Serialized model int8+zlib: `15940705 bytes`
- Code size: `46436 bytes`
- Total submission size int8+zlib: `15987141 bytes`

## Block Structure

Instead of standard residual additions, each block performs **spherical interpolation** toward attention/MLP targets using learned per-dimension gating scalars $\alpha_A, \alpha_M \in \mathbb{R}^d$:

```
h ← normalize(h + α_A ⊙ (normalize(attn(h)) − h))
h ← normalize(h + α_M ⊙ (normalize(mlp(h))  − h))
```

The $\alpha$ scalars are kept non-negative via `abs()` and act as a per-dimension learning rate on the sphere.

## Attention

Standard multi-head causal attention with RoPE, but with two changes:

- **Per-head Q/K normalization:** after RoPE, Q and K are L2-normalized along the head dimension and then rescaled by a learned per-dim scalar $s_{qk}$ (reshaped per-head).
- **Scale $\sqrt{d_k}$ instead of $1/\sqrt{d_k}$:** since Q, K are unit-norm, raw dot products lie in $[-1, 1]$; the conventional softmax temperature is inverted to give logits in $[-\sqrt{d_k}, \sqrt{d_k}]$.

V and the output projection are unconstrained linear maps (with unit-norm rows).

## MLP

SwiGLU with learned per-channel reparameterization scalars $s_u, s_v$ on the hidden activations:

```
u   = s_u · √d · gate(h)
v   = s_v · √d · up(h)
out = down(silu(u) * v)
```

## Embeddings & Output

- Token embeddings are unit-norm rows; the input hidden state is `normalize(tok_emb(x))`.
- The output uses a tied (or untied) embedding for logits, scaled by a **per-vocab-entry** learned scalar $s_z$ (initialized to $\sqrt{d}$), since both $h$ and embedding rows are unit-norm. Logits are then passed through a `tanh` softcap.
