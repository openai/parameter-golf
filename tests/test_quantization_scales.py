"""Offline tests for the PyTorch and MLX matrix export quantizers."""

import io
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_gpt


def quantize(values, backend):
    if backend == "torch":
        q, scales = train_gpt.quantize_float_tensor(torch.from_numpy(values))
        return q.numpy(), scales.numpy()
    import mlx.core as mx
    import train_gpt_mlx

    with mx.stream(mx.cpu):
        return train_gpt_mlx.quantize_float_array(mx.array(values))


@pytest.mark.parametrize("backend", ["torch", "mlx"])
@pytest.mark.parametrize("magnitude", [1e-5, 1e-3, 0.02, 0.5, 2.0, 100.0])
def test_matrix_scales_follow_each_rows_range(backend, magnitude):
    row = np.linspace(-1.0, 1.0, 513, dtype=np.float32)
    values = np.stack([row * magnitude, row * (magnitude / 4), np.zeros_like(row)])
    q, scales = quantize(values, backend)
    assert q.dtype == np.int8 and scales.dtype == np.float16
    assert q.flags.c_contiguous and scales.flags.c_contiguous
    assert np.isfinite(scales).all() and (scales > 0).all()
    # Independent reference to the serialized scale with a representable floor.
    peaks = np.quantile(np.abs(values), train_gpt.INT8_CLIP_Q, axis=1)
    floor = float(np.nextafter(np.float16(0), np.float16(1)))
    expected_scale = np.maximum(peaks / 127, floor).astype(np.float16)
    np.testing.assert_allclose(scales, expected_scale, rtol=0.001, atol=floor)
    restored = q.astype(np.float32) * scales.astype(np.float32)[:, None]
    # Rounding plus clipping at the endpoint are each bounded by one stored step.
    assert np.all(np.abs(restored - values) <= scales.astype(np.float32)[:, None] * 1.01)
    assert np.count_nonzero(q[0]) > 450
    assert np.all(q[2] == 0)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_empty_rows_and_tiny_finite_values(backend):
    for values in [
        np.empty((0, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.full((2, 3), 1e-9, dtype=np.float32),
    ]:
        q, scales = quantize(values, backend)
        assert q.shape == values.shape and scales.shape == values.shape[:1]
        assert np.isfinite(scales).all() and (scales > 0).all()
        assert np.all(q == 0)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_large_state_dictionary_roundtrip_keeps_small_matrix(backend):
    values = np.linspace(-0.003, 0.003, 257 * 257, dtype=np.float32).reshape(257, 257)
    assert values.size > train_gpt.INT8_KEEP_FLOAT_MAX_NUMEL
    if backend == "torch":
        state = {"matrix.weight": torch.from_numpy(values)}
        packed, stats = train_gpt.quantize_state_dict_int8(state)
        buffer = io.BytesIO()
        torch.save(packed, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        result = train_gpt.dequantize_state_dict_int8(loaded)["matrix.weight"].numpy()
    else:
        import mlx.core as mx
        import train_gpt_mlx

        with mx.stream(mx.cpu):
            packed, stats = train_gpt_mlx.quantize_state_dict_int8(
                {"matrix.weight": mx.array(values)}
            )
            result = np.array(train_gpt_mlx.dequantize_state_dict_int8(packed)["matrix.weight"])
    assert stats["num_float_tensors"] == 1
    assert packed["__quant_format__"] == "int8_clean_per_row_v1"
    assert packed["qmeta"]["matrix.weight"] == {"scheme": "per_row", "axis": 0}
    assert result.dtype == np.float32
    assert np.count_nonzero(result) > values.size * 0.95
    assert np.max(np.abs(result - values)) < 3e-5


@pytest.mark.parametrize("backend", ["torch", "mlx"])
@pytest.mark.parametrize("magnitude", [0.0, 0.002])
def test_vector_scale_behavior_is_unchanged(backend, magnitude):
    values = np.linspace(-magnitude, magnitude, 257, dtype=np.float32)
    q, scale = quantize(values, backend)
    assert scale.dtype == np.float32
    peak = float(np.quantile(np.abs(values), train_gpt.INT8_CLIP_Q))
    expected = np.float32(peak / 127 if peak > 0 else 1.0)
    np.testing.assert_allclose(scale, expected, rtol=1e-6)
    restored = q.astype(np.float32) * scale
    assert np.max(np.abs(restored - values)) <= float(scale) * 0.51
