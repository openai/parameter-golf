"""Test SwiGLU forward pass shapes and param count."""
import sys, os, types
sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault('flash_attn_interface', types.SimpleNamespace(flash_attn_func=None))
sys.modules.setdefault('sentencepiece', types.SimpleNamespace(SentencePieceProcessor=None))
sys.modules.setdefault('torch.distributed', types.SimpleNamespace())
sys.modules.setdefault('torch.nn.parallel', types.SimpleNamespace(DistributedDataParallel=None))
sys.modules.setdefault('zstandard', types.SimpleNamespace())
import torch
import train_gpt as tg

B, T, D, H = 2, 16, 768, 1536
swiglu = tg.SwiGLU(D, H)
x = torch.randn(B, T, D)
out = swiglu(x)
assert out.shape == (B, T, D), f"Expected ({B},{T},{D}), got {out.shape}"
n_params = sum(p.numel() for p in swiglu.parameters())
assert n_params == 3 * D * H, f"Expected {3*D*H} params, got {n_params}"
print(f"PASS: SwiGLU OK  params={n_params:,}")
