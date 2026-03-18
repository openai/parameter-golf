"""Tests for numel-gated fake quantization in QAT training."""
import mlx.core as mx
from train_gpt_mlx import (
    CastedLinear,
    INT8_KEEP_FLOAT_MAX_NUMEL,
    fake_quantize_per_row,
)


def test_fake_quantize_per_row_roundtrip():
    """fake_quantize_per_row returns a different tensor (not bitwise identical)."""
    w = mx.random.normal((128, 64))
    w_q = fake_quantize_per_row(w)
    mx.eval(w_q)
    assert w.shape == w_q.shape
    assert not mx.array_equal(w, w_q), "Quantized weight should differ from original"


def test_casted_linear_skips_small_weights():
    """CastedLinear with numel <= threshold should NOT fake-quantize during training."""
    # 64x64 = 4096 << 65536 threshold
    layer = CastedLinear(64, 64)
    assert layer.weight.size <= INT8_KEEP_FLOAT_MAX_NUMEL
    x = mx.random.normal((1, 4, 64))
    out_train = layer(x, training=True)
    out_eval = layer(x, training=False)
    mx.eval(out_train, out_eval)
    assert mx.allclose(out_train, out_eval, atol=1e-6), (
        "Small-weight CastedLinear should produce identical train/eval output"
    )


def test_casted_linear_quantizes_large_weights():
    """CastedLinear with numel > threshold SHOULD fake-quantize during training."""
    # 512x512 = 262144 > 65536 threshold
    layer = CastedLinear(512, 512)
    assert layer.weight.size > INT8_KEEP_FLOAT_MAX_NUMEL
    x = mx.random.normal((1, 4, 512))
    out_train = layer(x, training=True)
    out_eval = layer(x, training=False)
    mx.eval(out_train, out_eval)
    assert not mx.allclose(out_train, out_eval, atol=1e-6), (
        "Large-weight CastedLinear should produce different train/eval output"
    )


if __name__ == "__main__":
    test_fake_quantize_per_row_roundtrip()
    test_casted_linear_skips_small_weights()
    test_casted_linear_quantizes_large_weights()
    print("All tests passed")
