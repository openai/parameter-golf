"""Run this on an H100 pod to check FA3 availability."""
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
print()

# Check all possible FA paths
paths = [
    "flash_attn_interface",
    "flash_attn.flash_attn_interface",
    "flash_attn.flash_attn_func",
    "flash_attn",
    "flash_attn.flash_attn_triton",
]
for path in paths:
    try:
        mod = __import__(path, fromlist=["flash_attn_func"])
        funcs = [x for x in dir(mod) if "attn" in x.lower() and callable(getattr(mod, x, None))]
        print(f"  {path}: OK — functions: {funcs[:5]}")
    except ImportError as e:
        print(f"  {path}: MISSING — {e}")

print()
# Check if flash_attn.flash_attn_interface.flash_attn_func is the Hopper version
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    import inspect
    src = inspect.getsource(flash_attn_func)
    if "hopper" in src.lower() or "sm90" in src.lower() or "tma" in src.lower():
        print("flash_attn_func appears to be Hopper-optimized!")
    else:
        print(f"flash_attn_func source ({len(src)} chars) — checking for CUDA kernel calls...")
        # Check if it calls into C++ extension
        if "_flash_attn" in src or "flash_attn_cuda" in src:
            print("  -> Calls C++ CUDA extension (likely FA2/FA3 depending on build)")
        if "flash_attn_varlen" in src:
            print("  -> Has varlen support")
except Exception as e:
    print(f"Could not inspect flash_attn_func: {e}")

# Quick benchmark: 1000 iterations of attention
print("\n=== Quick Benchmark ===")
import time
B, H, S, D = 32, 8, 2048, 64
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    # Warmup
    for _ in range(10):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 100 * 1000
    print(f"flash_attn.flash_attn_interface: {t:.2f}ms/iter")
except Exception as e:
    print(f"flash_attn.flash_attn_interface: FAILED — {e}")

# Compare with SDPA
q2 = q.transpose(1, 2)
k2 = k.transpose(1, 2)
v2 = v.transpose(1, 2)
for _ in range(10):
    torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
torch.cuda.synchronize()
t = (time.perf_counter() - t0) / 100 * 1000
print(f"F.scaled_dot_product_attention: {t:.2f}ms/iter")
