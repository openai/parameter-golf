"""Tests for scripts/causal/statistical_analysis.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.causal.statistical_analysis import (
    analyse_raw_runs,
    compute_platform_transfer,
)


def _make_raw_runs(
    treatment_bpbs: list[float],
    control_bpbs: list[float],
    platform: str = "mlx",
    seeds: list[int] | None = None,
    errors: list[int] | None = None,
) -> dict:
    """Helper to build a raw_runs.json dict.

    *errors* is a list of seed indices (0-based) in treatment that should
    carry an ``error`` field, simulating partial failure.
    """
    if seeds is None:
        seeds = list(range(len(treatment_bpbs)))
    errors = errors or []
    treatment_results = []
    for i, (seed, bpb) in enumerate(zip(seeds, treatment_bpbs)):
        entry: dict = {"seed": seed, "val_bpb": bpb, "val_loss": 0.0, "wall_time_s": 60.0}
        if i in errors:
            entry["error"] = "timeout"
        treatment_results.append(entry)
    control_results = [
        {"seed": s, "val_bpb": b, "val_loss": 0.0, "wall_time_s": 60.0}
        for s, b in zip(seeds, control_bpbs)
    ]
    return {
        "platform": platform,
        "seeds": seeds,
        "treatment": {"config": {}, "results": treatment_results},
        "control": {"config": {}, "results": control_results},
    }


def _write_raw_runs(tmp: Path, raw: dict) -> Path:
    p = tmp / "raw_runs.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return p


# ---- T14 tests ----


def test_synthetic_known_effect():
    """Large, consistent effect across seeds => confirmed.

    With only 3 seeds the paired t-test needs a very tight spread to
    reach p < 0.01.  Use treatment and control values with identical
    paired differences so the t-test variance is zero-ish.
    """
    raw = _make_raw_runs(
        treatment_bpbs=[1.10, 1.11, 1.09],
        control_bpbs=[1.15, 1.16, 1.14],
        seeds=[42, 137, 256],
    )
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write_raw_runs(Path(tmp), raw)
        out = Path(tmp) / "ablation_results.json"
        result = analyse_raw_runs(str(inp), str(out))

        assert len(result["comparisons"]) == 1
        comp = result["comparisons"][0]
        # Treatment BPB is lower => mean_effect < 0 (improvement)
        assert comp["mean_effect"] < 0
        assert comp["decision"] == "confirmed"
        # Output file written
        assert out.exists()


def test_ci_contains_true_effect():
    """Bootstrap CI should contain the true mean difference."""
    treatment = [1.10, 1.11, 1.09]
    control = [1.15, 1.14, 1.16]
    true_diff = sum(t - c for t, c in zip(treatment, control)) / len(treatment)

    raw = _make_raw_runs(treatment, control, seeds=[42, 137, 256])
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write_raw_runs(Path(tmp), raw)
        out = Path(tmp) / "ablation_results.json"
        result = analyse_raw_runs(str(inp), str(out))

        comp = result["comparisons"][0]
        assert comp["ci_lo"] <= true_diff <= comp["ci_hi"]


def test_holm_bonferroni_adjusts():
    """Multiple p-values should be adjusted upward."""
    # Create two separate raw_runs files (simulating multiple comparisons)
    raw1 = _make_raw_runs([1.10, 1.11, 1.09], [1.15, 1.14, 1.16], seeds=[42, 137, 256])
    raw2 = _make_raw_runs([1.12, 1.13, 1.11], [1.15, 1.14, 1.16], seeds=[42, 137, 256])

    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "raw_runs_1.json"
        p1.write_text(json.dumps(raw1), encoding="utf-8")
        p2 = Path(tmp) / "raw_runs_2.json"
        p2.write_text(json.dumps(raw2), encoding="utf-8")
        out = Path(tmp) / "ablation_results.json"

        result = analyse_raw_runs(
            [str(p1), str(p2)], str(out)
        )

        assert len(result["comparisons"]) == 2
        for comp in result["comparisons"]:
            # Corrected p should be >= raw p
            assert comp["p_value_corrected"] >= comp["p_value"]


def test_decision_gate_null():
    """Effect < MDE => null."""
    # Nearly identical treatment and control
    raw = _make_raw_runs(
        [1.150, 1.150, 1.150],
        [1.151, 1.151, 1.151],
        seeds=[42, 137, 256],
    )
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write_raw_runs(Path(tmp), raw)
        out = Path(tmp) / "ablation_results.json"
        result = analyse_raw_runs(str(inp), str(out))
        assert result["comparisons"][0]["decision"] == "null"


def test_decision_gate_suggestive():
    """Effect >= MDE but p >= 0.01 => suggestive.

    With only 3 seeds and moderate variance, a real but noisy effect
    should yield p >= 0.01.
    """
    # Moderate effect with high variance across seeds
    raw = _make_raw_runs(
        [1.10, 1.16, 1.08],  # high variance
        [1.15, 1.12, 1.17],  # high variance
        seeds=[42, 137, 256],
    )
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write_raw_runs(Path(tmp), raw)
        out = Path(tmp) / "ablation_results.json"
        result = analyse_raw_runs(str(inp), str(out))
        comp = result["comparisons"][0]
        # The mean effect is (1.10-1.15 + 1.16-1.12 + 1.08-1.17)/3 = -0.033
        # But high cross-pair variance should make p >= 0.01
        assert comp["decision"] == "suggestive"


def test_partial_failure_handling():
    """Results with error field should be skipped."""
    raw = _make_raw_runs(
        treatment_bpbs=[1.10, 1.11, 1.09],
        control_bpbs=[1.15, 1.14, 1.16],
        seeds=[42, 137, 256],
        errors=[1],  # seed index 1 has error
    )
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write_raw_runs(Path(tmp), raw)
        out = Path(tmp) / "ablation_results.json"
        result = analyse_raw_runs(str(inp), str(out))

        comp = result["comparisons"][0]
        # Only 2 valid seeds
        assert comp["n_seeds"] == 2
        assert comp["decision"] in ("confirmed", "suggestive", "null")


def test_platform_transfer_coefficient():
    """Transfer coefficient computed when both platforms present."""
    mlx_effect = -0.05
    h100_effect = -0.04
    transfer = compute_platform_transfer(mlx_effect, h100_effect)
    assert transfer is not None
    assert "transfer_coefficient" in transfer
    assert transfer["transfer_coefficient"] == pytest.approx(
        h100_effect / mlx_effect, abs=1e-6
    )
    assert isinstance(transfer["divergence_flag"], bool)


def test_output_schema():
    """Output matches the I6 schema."""
    raw = _make_raw_runs([1.10, 1.11, 1.09], [1.15, 1.14, 1.16], seeds=[42, 137, 256])
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write_raw_runs(Path(tmp), raw)
        out = Path(tmp) / "ablation_results.json"
        result = analyse_raw_runs(str(inp), str(out))

        assert "comparisons" in result
        assert result["correction_method"] == "holm-bonferroni"
        assert result["alpha"] == 0.01
        assert result["mde"] == 0.002
        assert "platform_transfer" in result

        comp = result["comparisons"][0]
        for key in ("name", "platform", "n_seeds", "mean_effect", "ci_lo",
                     "ci_hi", "p_value", "p_value_corrected", "decision"):
            assert key in comp
