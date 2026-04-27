"""CPU tests for P0 MLP activation spec (LeakyReLU² vs ReLU²)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = (
    _ROOT
    / "records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
    / "mlp_activation_spec.py"
)


def _load_spec():
    spec = importlib.util.spec_from_file_location("mlp_activation_spec", _SPEC)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSP8192P0MLPActivation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_spec()

    def test_relu_squared_matches_nonnegative(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        y0 = self.mod.mlp_post_activation(x, 0.0)
        y05 = self.mod.mlp_post_activation(x, 0.5)
        self.assertTrue(torch.allclose(y0, y05))

    def test_leaky_differs_on_negative_fc(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 1.0], dtype=torch.float32)
        y0 = self.mod.mlp_post_activation(x, 0.0)
        y05 = self.mod.mlp_post_activation(x, 0.5)
        self.assertFalse(torch.allclose(y0, y05))
        self.assertTrue((y0[x < 0] == 0).all())
        self.assertTrue((y05[x < 0] != 0).all())


if __name__ == "__main__":
    unittest.main()
