# tests/test_ema.py
"""Tests for EMA weight averaging."""
import torch
import pytest


def ema_update(ema_state: dict, model_state: dict, decay: float = 0.997) -> dict:
    """Exact logic from train_gpt.py training loop."""
    with torch.no_grad():
        for name in ema_state:
            ema_state[name].mul_(decay).add_(
                model_state[name].detach().float(), alpha=1.0 - decay
            )
    return ema_state


class TestEMA:
    def test_ema_moves_toward_model(self):
        """EMA should move toward the current model weights."""
        ema = {"w": torch.zeros(10)}
        model = {"w": torch.ones(10)}
        ema_update(ema, model, decay=0.9)
        assert torch.allclose(ema["w"], torch.full((10,), 0.1)), \
            "EMA should be 0.1 after one step from 0 toward 1 with decay=0.9"

    def test_ema_converges_to_model(self):
        """After many steps, EMA should converge to model weights."""
        ema = {"w": torch.zeros(10)}
        model = {"w": torch.full((10,), 5.0)}
        for _ in range(10000):
            ema_update(ema, model, decay=0.997)
        assert torch.allclose(ema["w"], model["w"], atol=1e-2), \
            "EMA should converge to model weights after many steps"

    def test_ema_does_not_modify_model(self):
        """EMA update must not change model state dict."""
        ema = {"w": torch.zeros(10)}
        model = {"w": torch.ones(10)}
        orig = model["w"].clone()
        ema_update(ema, model, decay=0.997)
        assert torch.allclose(model["w"], orig), \
            "EMA update must not modify model weights"

    def test_ema_decay_997(self):
        """Verify exact formula with default decay=0.997."""
        ema = {"w": torch.tensor([2.0])}
        model = {"w": torch.tensor([4.0])}
        ema_update(ema, model, decay=0.997)
        expected = 0.997 * 2.0 + 0.003 * 4.0
        assert abs(ema["w"].item() - expected) < 1e-6, \
            f"EMA formula mismatch: expected {expected}, got {ema['w'].item()}"

    def test_ema_handles_multiple_keys(self):
        """EMA update works for all keys in state dict simultaneously."""
        ema = {"a": torch.zeros(5), "b": torch.zeros(3)}
        model = {"a": torch.ones(5), "b": torch.full((3,), 2.0)}
        ema_update(ema, model, decay=0.9)
        assert torch.allclose(ema["a"], torch.full((5,), 0.1))
        assert torch.allclose(ema["b"], torch.full((3,), 0.2))
