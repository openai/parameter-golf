from __future__ import annotations

import importlib
import os

import torch
from hypothesis import given
from hypothesis import strategies as st

from core.quant_core import (
    dequantize_state_dict_int8,
    quantize_float_tensor,
    quantize_state_dict_int8,
)


@st.composite
def float_tensor_case(draw):
    is_matrix = draw(st.booleans())
    if is_matrix:
        rows = draw(st.integers(min_value=1, max_value=6))
        cols = draw(st.integers(min_value=1, max_value=6))
        shape = (rows, cols)
        count = rows * cols
    else:
        count = draw(st.integers(min_value=1, max_value=32))
        shape = (count,)
    values = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=count,
            max_size=count,
        )
    )
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


@given(float_tensor_case())
def test_quantize_float_tensor_range_and_scales(t):
    q, s = quantize_float_tensor(t)
    assert q.dtype == torch.int8
    assert torch.all(q >= -127)
    assert torch.all(q <= 127)
    assert torch.all(s > 0)


def test_quantize_state_dict_roundtrip_preserves_keys_shapes_and_dtypes():
    state_dict = {
        "large.weight": torch.linspace(-0.25, 0.25, 70000, dtype=torch.float32).reshape(280, 250),
        "small.bias": torch.linspace(-1.0, 1.0, 128, dtype=torch.float32),
        "counter": torch.arange(8, dtype=torch.int64),
    }

    quant_obj, stats = quantize_state_dict_int8(state_dict)
    restored = dequantize_state_dict_int8(quant_obj)

    assert restored.keys() == state_dict.keys()
    assert stats["param_count"] == sum(t.numel() for t in state_dict.values())
    assert stats["baseline_tensor_bytes"] > 0
    assert stats["int8_payload_bytes"] > 0

    for name, original in state_dict.items():
        assert restored[name].shape == original.shape
        assert restored[name].dtype == original.dtype

    assert torch.equal(restored["counter"], state_dict["counter"])
    assert torch.allclose(restored["small.bias"], state_dict["small.bias"], atol=1e-3, rtol=0.0)

    q = quant_obj["quantized"]["large.weight"]
    scales = quant_obj["scales"]["large.weight"].to(torch.float32)
    max_scale = float(scales.max().item())
    max_error = float((restored["large.weight"] - state_dict["large.weight"]).abs().max().item())
    assert q.dtype == torch.int8
    assert max_error <= max_scale + 1e-6


def test_large_float_passthrough_override_keeps_selected_large_tensor(monkeypatch):
    monkeypatch.setenv("INT8_KEEP_LARGE_FLOAT_NAME_PATTERNS", "large.weight")
    import core.quant_core as quant_core

    quant_core = importlib.reload(quant_core)
    state_dict = {
        "large.weight": torch.linspace(-0.25, 0.25, 70000, dtype=torch.float32).reshape(280, 250),
        "other.weight": torch.linspace(-0.25, 0.25, 70000, dtype=torch.float32).reshape(280, 250),
    }

    quant_obj, stats = quant_core.quantize_state_dict_int8(state_dict)

    assert "large.weight" in quant_obj["passthrough"]
    assert "large.weight" not in quant_obj["quantized"]
    assert "other.weight" in quant_obj["quantized"]
    assert stats["num_large_float_passthrough_tensors"] == 1
    assert stats["large_float_passthrough_bytes"] == quant_core.tensor_nbytes(quant_obj["passthrough"]["large.weight"])
    assert quant_obj["passthrough"]["large.weight"].dtype == torch.float16
