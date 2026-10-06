"""CPU regression checks for finite large-gradient clipping in MLX."""

from pathlib import Path
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_gpt_mlx import clip_grad_tree


@pytest.mark.parametrize("magnitude", [1e10, 1e20, 1e25])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("limit", [1.0, 2.5])
def test_large_finite_gradients_are_clipped_without_erasing_the_update(magnitude, dtype, limit):
    with mx.stream(mx.cpu):
        tree = {
            "blocks": [{"weight": mx.array([[3 * magnitude, -4 * magnitude]], dtype=dtype)}],
            "bias": mx.array([magnitude], dtype=dtype),
        }
        before = {
            key: np.array(value.astype(mx.float32)).copy() for key, value in tree_flatten(tree)
        }
        reference_norm = np.sqrt(
            sum(np.sum(value.astype(np.float64) ** 2) for value in before.values())
        )
        result = clip_grad_tree(tree, limit)
        flattened = dict(tree_flatten(result))
        assert flattened.keys() == before.keys()
        for key, actual in flattened.items():
            values = np.array(actual.astype(mx.float32))
            expected = before[key].astype(np.float64) * (limit / (reference_norm + 1e-12))
            np.testing.assert_allclose(values, expected, rtol=0.015, atol=1e-6)
            assert np.isfinite(values).all() and np.count_nonzero(values) == values.size
            assert actual.dtype == dtype
        for key, original in tree_flatten(tree):
            np.testing.assert_array_equal(np.array(original.astype(mx.float32)), before[key])


@pytest.mark.parametrize("limit", [0.0, -1.0, 10.0])
def test_disabled_or_unnecessary_clipping_returns_original_tree(limit):
    with mx.stream(mx.cpu):
        tree = {"weight": mx.array([3.0, 4.0], dtype=mx.float32)}
        assert clip_grad_tree(tree, limit) is tree


def test_empty_and_zero_gradients_remain_unchanged():
    with mx.stream(mx.cpu):
        for tree in [{}, {"weight": mx.zeros((2, 3))}]:
            assert clip_grad_tree(tree, 1.0) is tree
