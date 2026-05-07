"""Hyperparameter env wiring for orchestrated record (SMOKE_MODE)."""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_GPT = _ROOT / "records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5/train_gpt.py"


def _load_orch_train():
    spec = importlib.util.spec_from_file_location("orch_train_smoke", _TRAIN_GPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOrchestratedSmokeHyperparams(unittest.TestCase):
    def test_smoke_mode_default_off(self):
        base = {k: v for k, v in os.environ.items() if k != "SMOKE_MODE"}
        with patch.dict(os.environ, base, clear=True):
            mod = _load_orch_train()
        self.assertFalse(mod.Hyperparameters.smoke_mode)

    def test_smoke_mode_env_on(self):
        with patch.dict(os.environ, {"SMOKE_MODE": "1"}, clear=False):
            mod = _load_orch_train()
        self.assertTrue(mod.Hyperparameters.smoke_mode)


if __name__ == "__main__":
    unittest.main()
