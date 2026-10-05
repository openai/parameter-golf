"""CPU regressions for the baseline rotary cache; no data or model downloads."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_gpt import Rotary, apply_rotary_emb


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [257, 1024])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_rotary_angles_survive_model_dtype_conversion(model_dtype, seq_len, warm_cache):
    dim, base = 64, 10000.0
    rotary = Rotary(dim, base)
    if warm_cache:
        rotary(128, torch.device("cpu"), torch.float32)
    rotary.to(dtype=model_dtype)
    cos, sin = rotary(seq_len, torch.device("cpu"), torch.float32)
    # Independent float64 construction of position * base**(-2*i/dim).
    positions = np.arange(seq_len, dtype=np.float64)
    frequencies = np.array([base ** (-2 * i / dim) for i in range(dim // 2)])
    angles = positions[:, None] * frequencies[None, :]
    expected_cos = torch.from_numpy(np.cos(angles)).float()[None, None]
    expected_sin = torch.from_numpy(np.sin(angles)).float()[None, None]
    torch.testing.assert_close(cos, expected_cos, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sin, expected_sin, atol=1e-4, rtol=1e-4)
    assert rotary.state_dict() == {}
    repeated = rotary(seq_len, torch.device("cpu"), torch.float32)
    assert repeated[0].data_ptr() == cos.data_ptr()
    assert repeated[1].data_ptr() == sin.data_ptr()


@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_neighboring_positions_remain_distinct_after_bfloat16_conversion(output_dtype):
    rotary = Rotary(16).bfloat16()
    cos, sin = rotary(1024, torch.device("cpu"), output_dtype)
    assert cos.dtype == output_dtype and sin.dtype == output_dtype
    assert not torch.equal(cos[:, :, 256], cos[:, :, 257])
    assert not torch.equal(sin[:, :, 256], sin[:, :, 257])
    assert float(cos[0, 0, 257, 0]) == pytest.approx(math.cos(257), abs=0.004)


def test_rotary_application_and_gradients_match_full_precision_reference():
    rotary = Rotary(16, base=5000).bfloat16()
    cos, sin = rotary(513, torch.device("cpu"), torch.float32)
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(1, 2, 513, 16, generator=generator, requires_grad=True)
    positions = torch.arange(513, dtype=torch.float64)
    frequencies = torch.tensor([5000 ** (-2 * i / 16) for i in range(8)], dtype=torch.float64)
    angles = positions[:, None] * frequencies[None, :]
    c, s = angles.cos().float()[None, None], angles.sin().float()[None, None]
    expected = torch.cat(
        (x[..., :8] * c + x[..., 8:] * s, -x[..., :8] * s + x[..., 8:] * c), dim=-1
    )
    actual = apply_rotary_emb(x, cos, sin)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    (actual_grad,) = torch.autograd.grad(actual.sum(), x, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected.sum(), x)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("compiled", [False, True])
def test_rotary_cache_stays_precise_under_autocast(compiled):
    rotary = Rotary(16).bfloat16()
    run = torch.compile(rotary, backend="eager", fullgraph=True) if compiled else rotary
    for length in (257, 513, 257):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            cos, sin = run(length, torch.device("cpu"), torch.float32)
        positions = np.arange(length, dtype=np.float64)
        frequencies = np.array([10000 ** (-2 * i / 16) for i in range(8)])
        angles = positions[:, None] * frequencies[None, :]
        torch.testing.assert_close(
            cos, torch.from_numpy(np.cos(angles)).float()[None, None], atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            sin, torch.from_numpy(np.sin(angles)).float()[None, None], atol=1e-4, rtol=1e-4
        )
