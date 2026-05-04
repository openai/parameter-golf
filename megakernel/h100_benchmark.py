"""
H100 Benchmark: Fused Mega-Kernels vs Baseline

Run on Thunder Compute 1x H100 PCIe:
    python3 megakernel/h100_benchmark.py

Tests:
  1. Kernel 1: FusedRMSNormMLP vs unfused RMSNorm + FusedLeakyReLUSquareMLP
  2. Kernel 2: FusedRMSNormQKV vs unfused RMSNorm + F.linear × 3
  3. Combined: Both kernels vs baseline in a simulated block forward pass

Expected on H100 (3.35 TB/s HBM3):
  Kernel 1 savings: ~75MB write eliminated per layer → ~0.5ms per forward pass (11 MLP layers)
  Kernel 2 savings: ~75MB write eliminated per layer → ~0.5ms per forward pass (11 attn layers)
  Combined: ~1-2ms per step = ~1.5-2.5% more optimizer steps
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Torch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print()

# Check TMA availability (H100 Hopper-only)
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    TMA = True
    print("TMA: AVAILABLE (H100 confirmed)")
except ImportError:
    TMA = False
    print("TMA: NOT AVAILABLE (not on H100)")
print()

# ─── Import kernels ───────────────────────────────────────────────────────────
from kernel1_rmsnorm_mlp import FusedRMSNormMLP
from kernel2_rmsnorm_qkv import FusedRMSNormQKVApply, fused_rmsnorm_qkv

DTYPE  = torch.bfloat16
DEVICE = "cuda"
REPS   = 500

def bench(fn, warmup=20, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000  # ms


# ─── Competition token count (FA3 packed batch on 8xH100) ────────────────────
# 8 GPUs × 512 sequences × 144 tokens/seq = 73728 tokens per GPU
M  = 73728  # tokens per GPU
K  = 512    # d_model
N  = 1536   # d_mlp (3× = 1536)
Nq = 512    # d_q (8 heads × 64 head_dim)
Nk = 256    # d_kv (4 KV heads × 64 head_dim)
Nv = 256

print(f"Token count per GPU: {M}")
print(f"Model dims: d={K}, d_mlp={N}, d_q={Nq}, d_kv={Nk}")
print()

torch.manual_seed(0)
x    = torch.randn(M, K,  dtype=DTYPE, device=DEVICE) * 0.1
up_w = torch.randn(N, K,  dtype=DTYPE, device=DEVICE) * 0.02
dn_w = torch.randn(K, N,  dtype=DTYPE, device=DEVICE) * 0.02
q_w  = torch.randn(Nq, K, dtype=DTYPE, device=DEVICE) * 0.02
k_w  = torch.randn(Nk, K, dtype=DTYPE, device=DEVICE) * 0.02
v_w  = torch.randn(Nv, K, dtype=DTYPE, device=DEVICE) * 0.02
scale = 1.0 / math.sqrt(3)
eps   = 1e-6


# ─── BENCHMARK 1: Kernel 1 (MLP) ─────────────────────────────────────────────
print("=" * 60)
print("KERNEL 1: Fused RMSNorm + MLP")
print("=" * 60)

def ref_mlp():
    x_n = F.rms_norm(x, (K,), weight=None, eps=eps) * scale
    h   = F.leaky_relu(F.linear(x_n, up_w), negative_slope=0.5)
    return F.linear(h * h, dn_w)

def fused_mlp():
    return FusedRMSNormMLP(x, up_w, dn_w, scale, eps)

t_ref   = bench(ref_mlp)
t_fused = bench(fused_mlp)
speedup = t_ref / t_fused
saving_11layers = (t_ref - t_fused) * 11 * 2  # 11 layers, fwd+bwd
print(f"  Baseline (RMSNorm + LeakyReLU²MLP): {t_ref:.3f} ms/call")
print(f"  Fused (K1):                          {t_fused:.3f} ms/call")
print(f"  Speedup: {speedup:.2f}x")
print(f"  Est. saving per step (11 MLP layers, fwd+bwd): {saving_11layers:.2f} ms")
print()


# ─── BENCHMARK 2: Kernel 2 (QKV) ─────────────────────────────────────────────
print("=" * 60)
print("KERNEL 2: Fused RMSNorm + QKV")
print("=" * 60)

def ref_qkv():
    x_n = F.rms_norm(x, (K,), weight=None, eps=eps) * scale
    q   = F.linear(x_n, q_w)
    k   = F.linear(x_n, k_w)
    v   = F.linear(x_n, v_w)
    return q, k, v

def fused_qkv():
    q, k, v, _ = fused_rmsnorm_qkv(x, q_w, k_w, v_w, scale=scale, eps=eps)
    return q, k, v

t_ref   = bench(ref_qkv)
t_fused = bench(fused_qkv)
speedup = t_ref / t_fused
saving_11layers = (t_ref - t_fused) * 11 * 2  # 11 attn layers, fwd+bwd
print(f"  Baseline (RMSNorm + Q+K+V linear):   {t_ref:.3f} ms/call")
print(f"  Fused (K2):                          {t_fused:.3f} ms/call")
print(f"  Speedup: {speedup:.2f}x")
print(f"  Est. saving per step (11 attn layers, fwd+bwd): {saving_11layers:.2f} ms")
print()


# ─── BENCHMARK 3: Combined (simulated Block forward) ─────────────────────────
print("=" * 60)
print("COMBINED: Simulated Block forward pass (attn + MLP)")
print("=" * 60)

# Simulated attention out + residual (stand-in for actual flash attention)
def make_attn_out():
    return torch.randn(M, K, dtype=DTYPE, device=DEVICE) * 0.01

def ref_block():
    # Attn: RMSNorm + Q+K+V
    x_n_a = F.rms_norm(x, (K,), weight=None, eps=eps) * scale
    q = F.linear(x_n_a, q_w)
    k = F.linear(x_n_a, k_w)
    v = F.linear(x_n_a, v_w)
    x_out = x + make_attn_out()  # residual
    # MLP: RMSNorm + up + act + down
    x_n_m = F.rms_norm(x_out, (K,), weight=None, eps=eps) * scale
    h     = F.leaky_relu(F.linear(x_n_m, up_w), negative_slope=0.5)
    out   = F.linear(h * h, dn_w)
    return x_out + out

def fused_block():
    # K2: fused attn RMSNorm + QKV
    q, k, v, _ = fused_rmsnorm_qkv(x, q_w, k_w, v_w, scale=scale, eps=eps)
    x_out = x + make_attn_out()
    # K1: fused MLP RMSNorm
    out   = FusedRMSNormMLP(x_out, up_w, dn_w, scale, eps)
    return x_out + out

t_ref   = bench(ref_block)
t_fused = bench(fused_block)
speedup = t_ref / t_fused
print(f"  Baseline block forward: {t_ref:.3f} ms")
print(f"  Fused block forward:    {t_fused:.3f} ms")
print(f"  Speedup: {speedup:.2f}x")
print()

# ─── MEMORY BANDWIDTH ANALYSIS ───────────────────────────────────────────────
print("=" * 60)
print("MEMORY BANDWIDTH ANALYSIS")
print("=" * 60)
props = torch.cuda.get_device_properties(0)
bus_bytes = props.memory_bus_width / 8  # bits → bytes
bw_hbm = (bus_bytes * props.memory_clock_rate * 1000 * 2) / 1e12  # DDR: × 2
token_bytes = M * K * 2  # bfloat16
bw_bytes = bw_hbm * 1e12  # TB/s → bytes/s
print(f"  Approx HBM bandwidth: {bw_hbm:.2f} TB/s")
print(f"  normed_x tensor size: {token_bytes / 1e6:.1f} MB")
print(f"  Eliminating 1 normed_x write: {token_bytes / bw_bytes * 1000:.4f} ms")
print(f"  Eliminated per step (22 layers, fwd+bwd): "
      f"{token_bytes * 22 * 2 / bw_bytes * 1000:.3f} ms")
print()

# ─── CORRECTNESS CHECK (sanity) ──────────────────────────────────────────────
print("=" * 60)
print("CORRECTNESS SANITY CHECK")
print("=" * 60)

x_n = F.rms_norm(x, (K,), weight=None, eps=eps) * scale
ref_out = F.linear(F.leaky_relu(F.linear(x_n, up_w), 0.5).square(), dn_w)
fus_out = FusedRMSNormMLP(x, up_w, dn_w, scale, eps)
err = (fus_out - ref_out).abs().max().item()
print(f"  K1 max abs error: {err:.2e} {'PASS' if err < 0.1 else 'FAIL'}")

q_r, k_r, v_r = ref_qkv()
q_f, k_f, v_f = fused_qkv()
eq = (q_f - q_r).abs().max().item()
print(f"  K2 Q max abs error: {eq:.2e} {'PASS' if eq < 0.05 else 'FAIL'}")

print()
print("Benchmark complete.")
