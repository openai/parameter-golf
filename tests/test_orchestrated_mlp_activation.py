"""CPU tests for orchestrated record MLP activation (ReLU² vs LeakyReLU²)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_GPT = _ROOT / "records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5/train_gpt.py"


def _load_orch_train():
    spec = importlib.util.spec_from_file_location("orch_train_gpt", _TRAIN_GPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOrchestratedMLPActivation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_orch_train()

    def test_mlp_relu_squared_shape(self):
        MLP = self.mod.MLP
        m = MLP(64, 2.0, leaky_relu_slope=0.0)
        x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
        y = m(x)
        self.assertEqual(y.shape, (2, 16, 64))
        self.assertTrue(torch.isfinite(y).all())

    def test_mlp_leaky_differs_from_relu_when_hidden_negative(self):
        MLP = self.mod.MLP
        m_relu = MLP(8, 2.0, leaky_relu_slope=0.0)
        m_leak = MLP(8, 2.0, leaky_relu_slope=0.5)
        with torch.no_grad():
            m_relu.fc.weight.fill_(1.0)
            m_relu.proj.weight.fill_(0.01)
            m_leak.fc.weight.fill_(1.0)
            m_leak.proj.weight.fill_(0.01)
        x = torch.full((1, 2, 8), -3.0)
        yr = m_relu(x)
        yl = m_leak(x)
        self.assertFalse(torch.allclose(yr, yl, rtol=1e-3, atol=1e-3))

    def test_mlp_leaky_gradient_through_negative(self):
        MLP = self.mod.MLP
        m = MLP(32, 2.0, leaky_relu_slope=0.5)
        x = torch.randn(2, 4, 32, requires_grad=True)
        y = m(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
