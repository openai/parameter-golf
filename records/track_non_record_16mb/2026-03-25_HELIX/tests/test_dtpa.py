"""Test DTPA shapes, param count, and basic differentiability."""
import sys, os, types
sys.path.insert(0, os.path.dirname(__file__))
# Patch flash_attn_interface to use F.scaled_dot_product_attention for CPU tests
import torch, torch.nn.functional as F

def _cpu_flash_attn(q, k, v, causal=False):
    # q,k,v in [B, T, H, D] format; convert to [B, H, T, D] for SDPA
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    # GQA: expand k,v to match q head count
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)  # back to [B, T, H, D]

import types
fa3_mod = types.SimpleNamespace(flash_attn_func=_cpu_flash_attn)
sys.modules['flash_attn_interface'] = fa3_mod
sys.modules.setdefault('sentencepiece', types.SimpleNamespace(SentencePieceProcessor=None))
sys.modules.setdefault('torch.distributed', types.SimpleNamespace())
sys.modules.setdefault('torch.nn.parallel', types.SimpleNamespace(DistributedDataParallel=None))
sys.modules.setdefault('zstandard', types.SimpleNamespace())

import train_gpt as tg

dim, n_heads, n_kv, rank, rope_dims = 64, 4, 2, 2, 8
B, T = 2, 16

dtpa = tg.DTPA(dim, n_heads, n_kv, rank, rope_dims, 10000.0, use_xsa=False, num_iterations=3)
h = torch.randn(B, T, dim, requires_grad=True)
x_res = torch.randn(B, T, dim)
out = dtpa(h, x_res, layer_r=0)
assert out.shape == (B, T, dim), f"Expected ({B},{T},{dim}), got {out.shape}"
out.sum().backward()
assert h.grad is not None, "No gradient on h"

# Test XSA path
dtpa_xsa = tg.DTPA(dim, n_heads, n_kv, rank, rope_dims, 10000.0, use_xsa=True, num_iterations=3)
out_xsa = dtpa_xsa(h.detach(), x_res, layer_r=2)  # last iteration triggers XSA
assert out_xsa.shape == (B, T, dim), f"XSA: Expected ({B},{T},{dim}), got {out_xsa.shape}"

# Param count sanity check (rough: should be much less than standard GQA)
n_dtpa = sum(p.numel() for p in dtpa.parameters())
n_std_gqa = dim*dim + dim*(dim//2)*2 + dim*dim  # Q + 2*(KV) + O at d=dim
assert n_dtpa < n_std_gqa, f"DTPA ({n_dtpa}) should be < standard GQA ({n_std_gqa})"
print(f"PASS: DTPA OK  params={n_dtpa}  std_gqa_params={n_std_gqa}")
