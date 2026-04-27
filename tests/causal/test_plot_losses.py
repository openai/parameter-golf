"""Tests for plot_losses.py (Task 5)."""
import json
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.causal.plot_losses import plot_loss_curves


def _mock_raw_runs(n_seeds=1, n_steps=10):
    """Create mock raw_runs dict with step_losses."""
    def _results(base_loss, seeds):
        return [
            {
                "seed": s,
                "val_bpb": base_loss - 0.5,
                "step_losses": [{"step": i, "train_loss": base_loss - i * 0.01} for i in range(n_steps)],
            }
            for s in seeds
        ]
    seeds = list(range(42, 42 + n_seeds))
    return {
        "treatment": {"results": _results(6.5, seeds)},
        "control": {"results": _results(7.0, seeds)},
    }


class TestPlotFromMockData:
    def test_produces_png(self, tmp_path):
        out = tmp_path / "curves.png"
        data = _mock_raw_runs(n_seeds=1, n_steps=10)
        plot_loss_curves(data, str(out))
        assert out.exists()
        assert out.stat().st_size > 0


class TestHandlesMissingStepLosses:
    def test_no_step_losses_no_crash(self, tmp_path):
        out = tmp_path / "empty.png"
        data = {
            "treatment": {"results": [{"seed": 42, "val_bpb": 1.2}]},
            "control": {"results": [{"seed": 42, "val_bpb": 1.3}]},
        }
        plot_loss_curves(data, str(out))
        assert out.exists()


class TestMultiSeedOverlay:
    def test_three_seeds(self, tmp_path):
        out = tmp_path / "multi.png"
        data = _mock_raw_runs(n_seeds=3, n_steps=20)
        plot_loss_curves(data, str(out))
        assert out.exists()
        assert out.stat().st_size > 0
