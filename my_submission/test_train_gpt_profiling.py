from __future__ import annotations

import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from my_submission import original_train_gpt as original_tg
from my_submission import train_gpt as tg
from my_submission.test_train_gpt import load_fused_qkv_from_original, project_qkv_old


def test_c_qkv_projection_is_faster_than_original_projection():
    threads_before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dim = 384
        num_heads = 6
        num_kv_heads = 2
        x = torch.randn(32, 128, dim)
        old_attn = original_tg.CausalSelfAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, rope_base=10000.0, qk_gain_init=1.5
        )
        fused_attn = tg.CausalSelfAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, rope_base=10000.0, qk_gain_init=1.5
        )
        load_fused_qkv_from_original(fused_attn, old_attn)

        def bench(fn, runs: int = 5, warmup: int = 10, iters: int = 40) -> float:
            with torch.inference_mode():
                for _ in range(warmup):
                    fn()
                samples: list[float] = []
                for _ in range(runs):
                    start = time.perf_counter()
                    for _ in range(iters):
                        fn()
                    samples.append((time.perf_counter() - start) / iters)
            return min(samples)

        old_s = bench(lambda: project_qkv_old(old_attn, x))
        fused_s = bench(lambda: fused_attn.project_qkv(x))
        assert fused_s < old_s
    finally:
        torch.set_num_threads(threads_before)
