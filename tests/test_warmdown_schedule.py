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

# noqa: E402
from train_gpt_mlx import Hyperparameters  # noqa: E402


def test_train_gpt_mlx_warmdown_iteration_boundary():
    # Case 1: Standard warmdown window (iterations=1000, warmdown_iters=200)
    args = Hyperparameters()
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


def test_train_gpt_mlx_warmdown_disabled():
    args = Hyperparameters()
    args.iterations = 1000
    args.warmdown_iters = 0
    args.max_wallclock_seconds = 0
    for step in [0, 500, 999, 1000, 1050]:
        assert args.lr_mul(step, 0.0) == 1.0


def test_train_gpt_mlx_warmdown_exceeds_iterations():
    # Warmdown equals or exceeds total iterations: decays from step 0
    args = Hyperparameters()
    args.iterations = 500
    args.warmdown_iters = 500
    args.max_wallclock_seconds = 0
    assert args.lr_mul(0, 0.0) == pytest.approx(1.0)
    assert args.lr_mul(250, 0.0) == pytest.approx(0.5)
    assert args.lr_mul(500, 0.0) == pytest.approx(0.0)
    assert args.lr_mul(550, 0.0) == pytest.approx(0.0)


def test_train_gpt_mlx_wallclock_expiration():
    args = Hyperparameters()
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


def test_train_gpt_step_schedule_logic():
    # Helper replicating train_gpt.py lr_mul closure
    def make_lr_mul(iterations: int, warmdown_iters: int, max_wallclock: float = 0.0):
        max_wallclock_ms = 1000.0 * max_wallclock if max_wallclock > 0 else None

        def lr_mul(step: int, elapsed_ms: float) -> float:
            if warmdown_iters <= 0:
                return 1.0
            if max_wallclock_ms is None:
                warmdown_start = max(iterations - warmdown_iters, 0)
                if step < warmdown_start:
                    return 1.0
                return max((iterations - step) / max(warmdown_iters, 1), 0.0)
            step_ms = elapsed_ms / max(step, 1)
            warmdown_ms = warmdown_iters * step_ms
            remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
            if remaining_ms <= warmdown_ms:
                return remaining_ms / max(warmdown_ms, 1e-9)
            return 1.0

        return lr_mul

    lr_mul = make_lr_mul(iterations=1000, warmdown_iters=200)
    assert lr_mul(0, 0.0) == 1.0
    assert lr_mul(799, 0.0) == 1.0
    assert lr_mul(800, 0.0) == pytest.approx(1.0)
    assert lr_mul(900, 0.0) == pytest.approx(0.5)
    assert lr_mul(999, 0.0) == pytest.approx(0.005)
    assert lr_mul(1000, 0.0) == pytest.approx(0.0)
    assert lr_mul(1001, 0.0) == pytest.approx(0.0)
    assert lr_mul(1100, 0.0) == pytest.approx(0.0)
