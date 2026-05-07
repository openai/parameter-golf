from __future__ import annotations

import math
import os
import time
from types import SimpleNamespace

import torch

import train_gpt as submission


def build_args() -> SimpleNamespace:
    return SimpleNamespace(
        kv_bits_mode=os.environ.get("KV_BITS_MODE", "balanced").strip().lower(),
        kv_group_size=int(os.environ.get("KV_GROUP_SIZE", "16")),
        kv_rotation_seed=int(os.environ.get("KV_ROTATION_SEED", "1234")),
        kv_cache_baseline=os.environ.get("KV_CACHE_BASELINE", "float").strip().lower(),
        kv_quant_backend=os.environ.get("KV_QUANT_BACKEND", "qjl").strip().lower(),
    )


def time_backend(backend, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, warmup: int, iters: int) -> dict[str, float]:
    cache = backend.append(None, k, v, max_len=k.size(2))
    score = backend.score(q, cache["k"])
    probs = torch.softmax(score.float(), dim=-1).to(dtype=q.dtype)
    out = backend.apply(probs, cache["v"])
    torch.cuda.synchronize()

    for _ in range(warmup):
        score = backend.score(q, cache["k"])
        probs = torch.softmax(score.float(), dim=-1).to(dtype=q.dtype)
        out = backend.apply(probs, cache["v"])
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        score = backend.score(q, cache["k"])
    torch.cuda.synchronize()
    score_ms = 1000.0 * (time.perf_counter() - t0) / iters

    probs = torch.softmax(score.float(), dim=-1).to(dtype=q.dtype)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = backend.apply(probs, cache["v"])
    torch.cuda.synchronize()
    apply_ms = 1000.0 * (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        score = backend.score(q, cache["k"])
        probs = torch.softmax(score.float(), dim=-1).to(dtype=q.dtype)
        out = backend.apply(probs, cache["v"])
    torch.cuda.synchronize()
    total_ms = 1000.0 * (time.perf_counter() - t0) / iters
    return {
        "score_ms": score_ms,
        "apply_ms": apply_ms,
        "total_ms": total_ms,
        "score": score.detach(),
        "out": out.detach(),
    }


def compare_pair(name_a: str, name_b: str, args, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    head_dim = q.size(-1)
    backend_a = submission.make_named_kv_backend(name_a, args, head_dim, q.size(1), k.size(1), q.device)
    backend_b = submission.make_named_kv_backend(name_b, args, head_dim, q.size(1), k.size(1), q.device)

    stats_a = time_backend(backend_a, q, k, v, warmup=30, iters=300)
    stats_b = time_backend(backend_b, q, k, v, warmup=30, iters=300)
    score_diff = float((stats_a["score"] - stats_b["score"]).abs().max().item())
    out_diff = float((stats_a["out"] - stats_b["out"]).abs().max().item())
    speedup = stats_a["total_ms"] / max(stats_b["total_ms"], 1e-9)
    print(
        f"{name_a:>12} total_ms={stats_a['total_ms']:.4f} "
        f"(score={stats_a['score_ms']:.4f} apply={stats_a['apply_ms']:.4f})"
    )
    print(
        f"{name_b:>12} total_ms={stats_b['total_ms']:.4f} "
        f"(score={stats_b['score_ms']:.4f} apply={stats_b['apply_ms']:.4f})"
    )
    print(f"{name_b:>12} speedup_vs_{name_a}={speedup:.2f}x score_max_abs={score_diff:.6e} out_max_abs={out_diff:.6e}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not submission.triton_is_available():
        raise RuntimeError("This benchmark requires Triton-backed kernels")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    batch = int(os.environ.get("BENCH_BATCH", "1"))
    num_heads = int(os.environ.get("BENCH_NUM_HEADS", "8"))
    num_kv_heads = int(os.environ.get("BENCH_NUM_KV_HEADS", "4"))
    seq_len = int(os.environ.get("BENCH_SEQ_LEN", "1024"))
    head_dim = int(os.environ.get("BENCH_HEAD_DIM", "64"))
    dtype = torch.float32

    if num_heads % num_kv_heads != 0:
        raise ValueError("BENCH_NUM_HEADS must be divisible by BENCH_NUM_KV_HEADS")

    q = torch.randn(batch, num_heads, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype) / math.sqrt(head_dim)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype) / math.sqrt(head_dim)
    args = build_args()

    print(f"device={torch.cuda.get_device_name(device)} seq_len={seq_len} head_dim={head_dim} kv_bits_mode={args.kv_bits_mode}")
    compare_pair("int8_backend", "int8_triton", args, q, k, v)
    compare_pair("qjl", "qjl_triton", args, q, k, v)


if __name__ == "__main__":
    main()
