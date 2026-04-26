"""
CPU smoke test: verify model builds, forward pass works, and quantization pipeline runs.
Does NOT require CUDA.
"""
import sys
import os
os.environ["WORLD_SIZE"] = "1"
os.environ["VOCAB_SIZE"] = "8192"

import torch
import numpy as np

# Patch: override CUDA requirement for CPU test
_orig_cuda_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False

print(f"Python {sys.version}")
print(f"PyTorch {torch.__version__}")

# Import model components
sys.path.insert(0, ".")
from train_gpt import (
    Hyperparameters, GPT, RMSNorm, CastedLinear, Muon, Rotary,
    apply_rotary_emb, restore_fp32_params, gptq_quantize_weight,
    _byte_shuffle, _byte_unshuffle, _compress, _decompress,
    classify_param,
)

print("\n--- Testing Model Instantiation ---")
h = Hyperparameters()
print(f"  vocab_size: {h.vocab_size}")
print(f"  num_layers: {h.num_layers}")
print(f"  model_dim: {h.model_dim}")
print(f"  num_heads: {h.num_heads}")
print(f"  num_kv_heads: {h.num_kv_heads}")
print(f"  mlp_mult: {h.mlp_mult}")
print(f"  num_loops: {h.num_loops} (layers {h.loop_start}-{h.loop_end})")
print(f"  parallel_residual_start: {h.parallel_residual_start}")
print(f"  qk_gain_init: {h.qk_gain_init}")

model = GPT(h).float()
num_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {num_params:,}")
print(f"  Encoder indices: {model.encoder_indices}")
print(f"  Decoder indices: {model.decoder_indices}")
print(f"  Num skip weights: {model.num_skip_weights}")
print(f"  Skip gates: {'yes' if model.skip_gates is not None else 'no'}")

# Check parallel residual flags
parallel_layers = [i for i, b in enumerate(model.blocks) if b.parallel]
xsa_layers = [i for i, b in enumerate(model.blocks) if b.attn.use_xsa]
print(f"  Parallel residual layers: {parallel_layers}")
print(f"  XSA layers: {xsa_layers}")

print("\n--- Testing Forward Pass ---")
batch_size = 2
seq_len = 64
x = torch.randint(0, h.vocab_size, (batch_size, seq_len))
y = torch.randint(0, h.vocab_size, (batch_size, seq_len))

# Test without looping
model.looping_active = False
loss_no_loop = model(x, y)
print(f"  Loss (no looping): {loss_no_loop.item():.4f}")

# Test with looping
model.looping_active = True
loss_loop = model(x, y)
print(f"  Loss (with looping): {loss_loop.item():.4f}")

# Test logits shape
logits = model.forward_logits(x)
print(f"  Logits shape: {logits.shape} (expected [{batch_size}, {seq_len}, {h.vocab_size}])")
assert logits.shape == (batch_size, seq_len, h.vocab_size), "Logits shape mismatch!"

print("\n--- Testing GPTQ Quantization ---")
# Create a synthetic weight + Hessian
w = torch.randn(256, 512) * 0.02
H = torch.eye(512) * 0.1 + torch.randn(512, 512) * 0.001
H = H @ H.T  # Make PSD

q, s = gptq_quantize_weight(w, H, clip_sigmas=12.85, clip_range=31)
print(f"  Quantized shape: {q.shape}, dtype: {q.dtype}")
print(f"  Scale shape: {s.shape}, dtype: {s.dtype}")
print(f"  Q range: [{q.min().item()}, {q.max().item()}]")
dequant = q.float() * s.float().unsqueeze(1)
mse = ((w - dequant) ** 2).mean().item()
print(f"  Dequant MSE: {mse:.6f}")

print("\n--- Testing Byte-Shuffle ---")
data = os.urandom(1000)
shuffled = _byte_shuffle(data)
unshuffled = _byte_unshuffle(shuffled)
assert data == unshuffled, "Byte-shuffle roundtrip FAILED!"
print(f"  Byte-shuffle roundtrip: OK")

print("\n--- Testing Compression ---")
import io
# Test with lzma (brotli requires separate install check)
test_data = os.urandom(10000)
for comp in ["lzma"]:
    compressed = _compress(test_data, comp)
    decompressed = _decompress(compressed, comp)
    assert test_data == decompressed, f"{comp} roundtrip FAILED!"
    print(f"  {comp}: {len(test_data)} -> {len(compressed)} ({len(compressed)/len(test_data)*100:.1f}%)")

try:
    import brotli
    compressed = _compress(test_data, "brotli")
    decompressed = _decompress(compressed, "brotli")
    assert test_data == decompressed, "brotli roundtrip FAILED!"
    print(f"  brotli: {len(test_data)} -> {len(compressed)} ({len(compressed)/len(test_data)*100:.1f}%)")
except ImportError:
    print("  brotli: not installed (will use lzma)")

print("\n--- Estimated Submission Size ---")
# Estimate model size after int6 GPTQ + brotli
param_bytes_raw = num_params  # 1 byte per param (int8 storage)
est_compressed = int(param_bytes_raw * 0.42)  # ~42% compression ratio
code_size = 19179  # From our build_submission.py test
est_total = est_compressed + code_size
print(f"  Params: {num_params:,}")
print(f"  Est. compressed model: {est_compressed:,} bytes ({est_compressed/1e6:.2f} MB)")
print(f"  Code bytes: {code_size:,}")
print(f"  Est. total: {est_total:,} bytes ({est_total/1e6:.2f} MB)")
print(f"  Budget remaining: {16_000_000 - est_total:,} bytes")
if est_total > 16_000_000:
    print(f"  OVER BUDGET by {est_total - 16_000_000:,} bytes!")
else:
    print(f"  Under 16MB budget OK")

print("\n=== ALL SMOKE TESTS PASSED ===")
