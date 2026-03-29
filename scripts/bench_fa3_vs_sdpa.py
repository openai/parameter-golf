"""Benchmark: native SDPA flash backend vs explicit FA3 on H100.

Matches the anchor model dimensions:
  model_dim=512, num_heads=8, num_kv_heads=4, head_dim=64, seq_len=2048

Usage (26.03 container — native SDPA only):
  python bench_fa3_vs_sdpa.py --sdpa-only

Usage (25.02 container — both):
  pip install flash_attn_3-3.0.0-... && python bench_fa3_vs_sdpa.py
"""

import argparse
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# Anchor model dimensions
B = 16          # batch size (typical micro-batch)
T = 2048        # sequence length
H = 8           # num_heads
Hkv = 4         # num_kv_heads
D = 64          # head_dim (512 / 8)

WARMUP = 50
ITERS = 200


def bench_sdpa_flash(q_bhtd, k_bhtd, v_bhtd, warmup=WARMUP, iters=ITERS):
    """Benchmark F.scaled_dot_product_attention with flash backend forced."""
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for _ in range(warmup):
            F.scaled_dot_product_attention(q_bhtd, k_bhtd, v_bhtd, is_causal=True,
                                           enable_gqa=(Hkv != H))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        F.scaled_dot_product_attention(q_bhtd, k_bhtd, v_bhtd, is_causal=True,
                                       enable_gqa=(Hkv != H))
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def bench_fa3_direct(q_bthd, k_bthd, v_bthd, warmup=WARMUP, iters=ITERS):
    """Benchmark flash_attn_3_func with (B, T, H, D) layout."""
    from flash_attn_interface import flash_attn_func
    for _ in range(warmup):
        flash_attn_func(q_bthd, k_bthd, v_bthd, causal=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        flash_attn_func(q_bthd, k_bthd, v_bthd, causal=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdpa-only", action="store_true",
                        help="Only benchmark SDPA (for containers without flash_attn_interface)")
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: B={B}, T={T}, H={H}, Hkv={Hkv}, D={D}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print()

    # SDPA layout: (B, H, T, D)
    q_bhtd = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k_bhtd = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.bfloat16)
    v_bhtd = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.bfloat16)

    # FA3 layout: (B, T, H, D)
    q_bthd = q_bhtd.transpose(1, 2).contiguous()
    k_bthd = k_bhtd.transpose(1, 2).contiguous()
    v_bthd = v_bhtd.transpose(1, 2).contiguous()

    # --- SDPA Flash ---
    ms_sdpa = bench_sdpa_flash(q_bhtd, k_bhtd, v_bhtd)
    print(f"SDPA flash:   {ms_sdpa:.3f} ms/iter")

    # --- FA3 Direct ---
    if not args.sdpa_only:
        try:
            ms_fa3 = bench_fa3_direct(q_bthd, k_bthd, v_bthd)
            print(f"FA3 direct:   {ms_fa3:.3f} ms/iter")
            print()
            speedup = ms_sdpa / ms_fa3
            faster = "FA3" if speedup > 1 else "SDPA"
            ratio = max(speedup, 1 / speedup)
            print(f"Winner: {faster} ({ratio:.2f}x)")
        except ImportError:
            print("flash_attn_interface not available — skipping FA3 benchmark")
            print("Run with --sdpa-only or pip install the FA3 wheel first")
    else:
        print("\n(--sdpa-only: FA3 benchmark skipped)")


if __name__ == "__main__":
    main()
