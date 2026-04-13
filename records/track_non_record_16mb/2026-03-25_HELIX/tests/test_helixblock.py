"""Test HELIXBlock forward and backward."""
import sys, os, types, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))

def _cpu_fa3(q, k, v, causal=False):
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)
sys.modules['flash_attn_interface'] = types.SimpleNamespace(flash_attn_func=_cpu_fa3)
sys.modules.setdefault('sentencepiece', types.SimpleNamespace(SentencePieceProcessor=None))
sys.modules.setdefault('torch.distributed', types.SimpleNamespace())
sys.modules.setdefault('torch.nn.parallel', types.SimpleNamespace(DistributedDataParallel=None))
sys.modules.setdefault('zstandard', types.SimpleNamespace())

import train_gpt as tg

dim, n_heads, n_kv, rank, rope_dims = 64, 4, 2, 2, 8
B, T, R = 2, 16, 3

block = tg.HELIXBlock(dim, n_heads, n_kv, rank, 128, rope_dims, 10000.0,
                       use_xsa=False, num_iterations=R, block_idx=0)
x  = torch.randn(B, T, dim, requires_grad=True)
x0 = torch.randn(B, T, dim)

for r in range(R):
    out = block(x, x0, r)
    assert out.shape == (B, T, dim), f"r={r}: expected ({B},{T},{dim}), got {out.shape}"

out.sum().backward()
assert x.grad is not None, "No gradient on x"

# Check per-iteration params exist
assert len(block.iter_attn_scale) == R
assert len(block.iter_mlp_scale) == R
assert len(block.iter_resid_mix) == R
print(f"PASS: HELIXBlock OK  n_iters={R}")
