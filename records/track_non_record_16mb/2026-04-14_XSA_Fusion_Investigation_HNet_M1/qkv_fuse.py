"""QKV weight-stacking patch for the SP8192 stack's `CausalSelfAttention`.

Stacks `c_q / c_k / c_v` into a single `c_qkv` linear of shape
    (num_heads * head_dim + 2 * num_kv_heads * head_dim,  dim)
and rebinds the attention module's `forward` to split/reshape the stacked output.

Numerical parity: forward-output at init is bit-identical to the 3-linear
baseline (we verified 0.000e+00 max elementwise error at multiple shapes).

Caveat: NOT a numerically-equivalent systems-only change under Muon. See the
"Muon interaction" note below. This file is shipped as reference / for anyone
following up — the training pilot showed that Inductor already fuses these three
linears when compiled at model scope, so there is no step-time benefit in practice.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def fuse_qkv_weights(attn, CastedLinear):
    """Replace attn.c_q / c_k / c_v with a single c_qkv of stacked weights.

    Preserves dtype and device of the original weights. Must be called with
    `CastedLinear` from the same module namespace as `attn` (because the rebound
    forward references its `__call__`).
    """
    dim = attn.c_q.weight.size(1)
    q_dim  = attn.num_heads    * attn.head_dim
    kv_dim = attn.num_kv_heads * attn.head_dim
    out_dim = q_dim + 2 * kv_dim

    c_qkv = CastedLinear(dim, out_dim, bias=False)
    c_qkv = c_qkv.to(attn.c_q.weight.device).to(attn.c_q.weight.dtype)
    with torch.no_grad():
        c_qkv.weight.copy_(torch.cat(
            [attn.c_q.weight, attn.c_k.weight, attn.c_v.weight],
            dim=0,
        ))

    attn.c_qkv = c_qkv
    del attn.c_q, attn.c_k, attn.c_v


def make_fused_qkv_forward(attn_backend, apply_rotary_emb, xsa_fn=None):
    """Build a replacement `forward` for a fused-QKV CausalSelfAttention.

    Args:
        attn_backend: callable with signature `(q, k, v, causal=...) -> y`
            accepting `(B, S, H, D)` layout for q,k,v (matches FA3's API).
        apply_rotary_emb: the baseline's partial-RoPE function.
        xsa_fn: optional replacement for `self._xsa_efficient(y, v)`. If None
            the instance's existing `_xsa_efficient` is called.

    Returns a function suitable for `types.MethodType(fn, attn_instance)`.
    """
    def forward(self, x):
        bsz, seqlen, _ = x.shape
        qkv = self.c_qkv(x)
        q_dim  = self.num_heads    * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.reshape(bsz, seqlen, self.num_heads,    self.head_dim)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = attn_backend(q, k, v, causal=True)
        if self.use_xsa:
            y = xsa_fn(y, v) if xsa_fn is not None else self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, y.size(-2) * y.size(-1))
        return self.proj(y)
    return forward


def patch_attn(attn, CastedLinear, attn_backend, apply_rotary_emb, xsa_fn=None):
    """One-shot convenience: fuse weights and rebind forward on an attn instance."""
    fuse_qkv_weights(attn, CastedLinear)
    attn.forward = types.MethodType(
        make_fused_qkv_forward(attn_backend, apply_rotary_emb, xsa_fn),
        attn,
    )


# Muon interaction (important):
# ---------------------------------------------------------------------------
# The baseline Muon optimizer runs the Newton-Schulz-5 polynomial on each 2-D
# weight matrix's gradient independently. With 3 separate linears (c_q, c_k, c_v),
# NS5 orthogonalizes three independent gradient matrices. With a fused c_qkv of
# shape (D + 2*D_kv, D), NS5 orthogonalizes the *joint* gradient matrix — a
# different spectrum in general, and therefore a different effective update.
# The forward output is bit-identical at init, but the training trajectories
# diverge. In our 200-step pilot we observed ~0.01 nats train_loss drift at step
# 200 (bf16), which is within the noise floor but IS a real effect.
#
# A correct systems-only fused-QKV would need one of:
#   (a) split the c_qkv gradient back into 3 slices and NS5 each independently,
#       then reassemble (matches baseline dynamics bit-for-bit);
#   (b) reformulate Muon so NS5 respects a block-diagonal / Kronecker structure
#       on the stacked weight;
#   (c) accept the divergence and claim the fused version as a slightly different
#       model, requiring a full 3-seed mean for the record threshold.
# Neither (a) nor (b) is implemented here.
