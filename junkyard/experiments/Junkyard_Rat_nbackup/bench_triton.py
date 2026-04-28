#!/usr/bin/env python3
"""
Junkyard_Rat — bench_triton.py
Bench the exact LeakyReLU^2 MLP path used by JR-01/JR-02.

Run on a single GPU:
    python experiments/Junkyard_Rat/bench_triton.py
"""
import time

import torch
import torch.nn.functional as F
from torch import nn


class RatRodBankMLP(nn.Module):
    def __init__(self, dim: int = 512, mlp_mult: float = 3.0):
        super().__init__()
        hidden = int(dim * mlp_mult)
        self.up_w = nn.Parameter(torch.randn(hidden, dim, dtype=torch.float32) * 0.02)
        self.down_w = nn.Parameter(torch.randn(dim, hidden, dtype=torch.float32) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(F.linear(x, self.up_w.to(x.dtype)), negative_slope=0.5)
        return F.linear(x.square(), self.down_w.to(x.dtype))


def bench(fn, x: torch.Tensor, label: str, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        torch.compiler.cudagraph_mark_step_begin()
        y = fn(x)
        y.sum().backward()
        x.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.compiler.cudagraph_mark_step_begin()
        y = fn(x)
        y.sum().backward()
        x.grad = None
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters
    print(f"{label:<30} {ms:8.3f} ms")
    return ms


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = "cuda"
    dtype = torch.bfloat16
    batch, seq_len, dim = 48, 2048, 512

    print("=" * 60)
    print("Junkyard_Rat — JR-02 Triton Candidate Bench")
    print(f"device={torch.cuda.get_device_name(0)} dtype={dtype} shape=({batch}, {seq_len}, {dim})")
    print("=" * 60)

    eager = RatRodBankMLP(dim=dim).to(device)
    compiled = RatRodBankMLP(dim=dim).to(device)
    compiled.load_state_dict(eager.state_dict())

    x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype, requires_grad=True)

    eager_ms = bench(lambda inp: eager(inp), x.detach().requires_grad_(True), "eager")
    compiled_fn = torch.compile(compiled, dynamic=False, fullgraph=False, mode="max-autotune")
    x_probe = x.detach().requires_grad_(True)
    torch.compiler.cudagraph_mark_step_begin()
    compiled_fn(x_probe).sum().backward()
    x_probe.grad = None
    torch.cuda.synchronize()
    tuned_ms = bench(lambda inp: compiled_fn(inp), x.detach().requires_grad_(True), "compile(max-autotune)")

    speedup = eager_ms / tuned_ms
    print("-" * 60)
    print(f"speedup: {speedup:.3f}x")
    if speedup >= 1.10:
        print("signal: proceed with JR-02 full run")
    else:
        print("signal: weak, JR-02 may not justify full run")


if __name__ == "__main__":
    main()
