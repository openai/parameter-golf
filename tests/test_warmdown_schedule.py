import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mock mlx on platforms without Apple Silicon MLX installed
for mod in ["mlx", "mlx.core", "mlx.nn", "mlx.optimizers", "mlx.utils"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from train_gpt import Hyperparameters as PyTorchHyperparameters  # noqa: E402
from train_gpt_mlx import Hyperparameters as MLXHyperparameters  # noqa: E402

ALL_HYPERPARAMETERS = [
    pytest.param(PyTorchHyperparameters, id="pytorch"),
    pytest.param(MLXHyperparameters, id="mlx"),
]


@pytest.mark.parametrize("hyperparameters_cls", ALL_HYPERPARAMETERS)
def test_warmdown_iteration_boundary(hyperparameters_cls):
    # Case 1: Standard warmdown window (iterations=1000, warmdown_iters=200)
    args = hyperparameters_cls()
    args.iterations = 1000
    args.warmdown_iters = 200
    args.max_wallclock_seconds = 0

    # Before warmdown
    assert args.lr_mul(0, 0.0) == 1.0
    assert args.lr_mul(500, 0.0) == 1.0
    assert args.lr_mul(799, 0.0) == 1.0

    # Warmdown start
    assert args.lr_mul(800, 0.0) == pytest.approx(1.0)

    # Intermediate points in warmdown
    assert args.lr_mul(900, 0.0) == pytest.approx(0.5)
    assert args.lr_mul(950, 0.0) == pytest.approx(0.25)
    assert args.lr_mul(999, 0.0) == pytest.approx(0.005)

    # At iteration boundary (previously jumped back to 1.0)
    assert args.lr_mul(1000, 0.0) == pytest.approx(0.0)

    # Beyond iteration boundary
    assert args.lr_mul(1001, 0.0) == pytest.approx(0.0)
    assert args.lr_mul(1500, 0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("hyperparameters_cls", ALL_HYPERPARAMETERS)
def test_warmdown_disabled(hyperparameters_cls):
    args = hyperparameters_cls()
    args.iterations = 1000
    args.warmdown_iters = 0
    args.max_wallclock_seconds = 0
    for step in [0, 500, 999, 1000, 1050]:
        assert args.lr_mul(step, 0.0) == 1.0


@pytest.mark.parametrize("hyperparameters_cls", ALL_HYPERPARAMETERS)
def test_warmdown_exceeds_iterations(hyperparameters_cls):
    # Warmdown equals or exceeds total iterations: decays from step 0
    args = hyperparameters_cls()
    args.iterations = 500
    args.warmdown_iters = 500
    args.max_wallclock_seconds = 0
    assert args.lr_mul(0, 0.0) == pytest.approx(1.0)
    assert args.lr_mul(250, 0.0) == pytest.approx(0.5)
    assert args.lr_mul(500, 0.0) == pytest.approx(0.0)
    assert args.lr_mul(550, 0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("hyperparameters_cls", ALL_HYPERPARAMETERS)
def test_wallclock_expiration(hyperparameters_cls):
    args = hyperparameters_cls()
    args.iterations = 10000
    args.warmdown_iters = 200
    args.max_wallclock_seconds = 600
    # 100 steps took 500s -> step_ms = 5000ms
    # warmdown_ms = 200 * 5000 = 1,000,000ms
    # remaining_ms = 600,000 - 500,000 = 100,000ms <= warmdown_ms
    assert args.lr_mul(100, 500_000.0) == pytest.approx(100_000 / 1_000_000)

    # Wallclock expired: remaining_ms = 0
    assert args.lr_mul(120, 600_000.0) == pytest.approx(0.0)
    assert args.lr_mul(125, 650_000.0) == pytest.approx(0.0)
