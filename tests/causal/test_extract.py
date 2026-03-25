"""Tests for scripts/causal/extract_interventions.py (C2 — Record Parser).

Covers:
- Format A parser (Change/Impact table)
- Format B parser (Base/This comparison)
- Format C fallback (prose only)
- field_coverage computation
- --append-experiment mode with mock raw_runs.json
- Unknown format fallback
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures: actual README content from the three canonical records
# ---------------------------------------------------------------------------

FORMAT_A_README = """\
## Record: 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 (val_bpb: 1.1233)

**val_bpb: 1.1233** (sliding window stride=64, 3-seed mean) | **15.55 MB** (mean) | 8xH100 SXM, 600s

### Key Innovations Over PR #374

Two novel post-training optimizations plus training hyperparameter tuning on top of PR #374's architecture:

| Change | PR #374 | This | Impact |
|--------|---------|------|--------|
| **GPTQ-lite** | Fixed clip (row max) | 5 clip percentiles per row, pick min MSE | -0.0006 BPB (zero training cost) |
| **EMA** | None (Tight SWA only) | EMA decay=0.997 every step | -0.0006 BPB (smoother averaging) |
| **Warmdown** | 3000 | 3500 | -0.0002 BPB |
| **Late QAT threshold** | 0.1 | 0.15 | -0.0001 BPB (earlier fake quant, smaller quant gap) |
| **Total** | **1.1246** | **1.1233** | **-0.0013 BPB** |
"""

FORMAT_A_SUBMISSION = {
    "author": "Tianhao Wu",
    "github_id": "signalrush",
    "name": "Record: 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15",
    "blurb": "EMA(0.997) weight averaging + GPTQ-lite optimal clip percentile search + warmdown=3500 + Late QAT threshold=0.15, built on PR#374 stack.",
    "date": "2026-03-22T00:00:00Z",
    "val_loss": 1.89576235,
    "val_bpb": 1.12278022,
    "bytes_total": 15555017,
}

FORMAT_B_README = """\
## Record: 11L Partial RoPE + LN Scale + EMA + XSA4 (val_bpb: 1.1248)

**val_bpb = 1.1248** (sliding window, stride=64) | **15.6 MB** artifact | 8xH100 SXM, 600s

Previous: [PR #70](https://github.com/openai/parameter-golf/pull/70) (9L, 1.1659) -> [PR #287](https://github.com/openai/parameter-golf/pull/287) (11L, 1.1271) -> this

### Changes from PR #287

| | [PR #287](https://github.com/openai/parameter-golf/pull/287) | This |
|---|---|---|
| val_bpb (sliding s64) | 1.1271 | **1.1248** |
| Partial RoPE | None (full 64d) | 16 of 64 dims |
| LN Scale | None | 1/sqrt(layer_idx+1) |
| Artifact | 15.5 MB | 15.6 MB |
| Everything else | Same | Same |

### What's new

1. **Partial RoPE (16 of 64 dims)**. Rotary position embeddings applied to only the first 16 of 64 head dimensions.

2. **LN Scale**. RMSNorm outputs are scaled by 1/sqrt(layer_idx+1), damping deeper layers' contributions.

### Note on Late QAT

The submitted code includes a Late QAT flag but it was inactive due to torch.compile constant-folding.
"""

FORMAT_B_SUBMISSION = {
    "author": "Jack Princz",
    "github_id": "jfprincz",
    "name": "Record: 11L Partial RoPE + LN Scale + EMA + XSA4",
    "blurb": "11 layers with Partial RoPE (16 of 64 dims), LN Scale (1/sqrt(l+1)), EMA weight averaging.",
    "date": "2026-03-21T06:00:00Z",
    "val_loss": 1.89924867,
    "val_bpb": 1.12484502,
    "bytes_total": 15612308,
}

FORMAT_C_README = """\
This record captures the `Simple Baseline`.

Trainer changes in this snapshot:
- current repository `train_gpt.py` snapshot copied into the record folder
- published `fineweb10B_sp1024` dataset and tokenizer loaded from the new Hugging Face export
- 10-minute wallclock cap on `8xH100`
- periodic validation every `200` steps on the full `fineweb_val_*` split

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
"""

FORMAT_C_SUBMISSION = {
    "author": "Baseline",
    "github_id": "openai",
    "name": "Naive Baseline",
    "blurb": "SP-1024 9x512 KV4 run on pgut1 using the published Hugging Face fineweb10B_sp1024 export.",
    "date": "2026-03-18T14:56:29Z",
    "val_loss": 2.07269931,
    "val_bpb": 1.2243657,
    "bytes_total": 15863489,
}


# ---------------------------------------------------------------------------
# Helper: create a temporary record directory
# ---------------------------------------------------------------------------

def _make_record(tmp: Path, name: str, readme: str, submission: dict) -> Path:
    """Create a mock record directory with README.md and submission.json."""
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "README.md").write_text(readme, encoding="utf-8")
    (d / "submission.json").write_text(json.dumps(submission), encoding="utf-8")
    return d


@pytest.fixture()
def records_dir(tmp_path: Path) -> Path:
    """Create a temp records directory with one record of each format."""
    base = tmp_path / "records"
    base.mkdir()
    _make_record(base, "2026-03-22_FormatA_1.1233", FORMAT_A_README, FORMAT_A_SUBMISSION)
    _make_record(base, "2026-03-21_FormatB_1.1248", FORMAT_B_README, FORMAT_B_SUBMISSION)
    _make_record(base, "2026-03-17_FormatC", FORMAT_C_README, FORMAT_C_SUBMISSION)
    return base


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import importlib
import sys


def _import_extract():
    """Import extract_interventions module."""
    mod_path = Path(__file__).resolve().parents[2] / "scripts" / "causal"
    if str(mod_path.parent) not in sys.path:
        sys.path.insert(0, str(mod_path.parent))
    # Ensure the causal package is importable
    spec = importlib.util.spec_from_file_location(
        "causal.extract_interventions",
        mod_path / "extract_interventions.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def extract():
    return _import_extract()


# ===================================================================
# Tests
# ===================================================================


class TestFormatAParsing:
    """Format A: Change/Impact ablation table."""

    def test_detects_format_a(self, extract):
        result = extract.detect_format(FORMAT_A_README)
        assert result == "A"

    def test_parses_interventions(self, extract):
        interventions = extract.parse_format_a(FORMAT_A_README)
        names = [i["name"] for i in interventions]
        assert "GPTQ-lite" in names
        assert "EMA" in names
        assert "Warmdown" in names
        assert "Late QAT threshold" in names
        # Should NOT include the "Total" summary row
        assert not any("total" in n.lower() for n in names)

    def test_delta_bpb_values(self, extract):
        interventions = extract.parse_format_a(FORMAT_A_README)
        deltas = {i["name"]: i["delta_bpb"] for i in interventions}
        assert deltas["GPTQ-lite"] == pytest.approx(-0.0006, abs=1e-5)
        assert deltas["EMA"] == pytest.approx(-0.0006, abs=1e-5)
        assert deltas["Warmdown"] == pytest.approx(-0.0002, abs=1e-5)

    def test_delta_source_is_ablation_table(self, extract):
        interventions = extract.parse_format_a(FORMAT_A_README)
        for i in interventions:
            assert i["delta_source"] == "ablation_table"

    def test_base_bpb_extracted(self, extract):
        base_bpb = extract.extract_base_bpb_format_a(FORMAT_A_README)
        # Total row shows Base=1.1246
        assert base_bpb == pytest.approx(1.1246, abs=1e-4)


class TestFormatBParsing:
    """Format B: Base/This comparison table."""

    def test_detects_format_b(self, extract):
        result = extract.detect_format(FORMAT_B_README)
        assert result == "B"

    def test_parses_interventions(self, extract):
        interventions = extract.parse_format_b(FORMAT_B_README)
        names = [i["name"] for i in interventions]
        assert "Partial RoPE" in names
        assert "LN Scale" in names

    def test_base_bpb_from_table(self, extract):
        base_bpb = extract.extract_base_bpb_format_b(FORMAT_B_README)
        assert base_bpb == pytest.approx(1.1271, abs=1e-4)

    def test_delta_source_is_readme_prose(self, extract):
        interventions = extract.parse_format_b(FORMAT_B_README)
        for i in interventions:
            assert i["delta_source"] == "readme_prose"


class TestFormatCParsing:
    """Format C: prose only (no tables)."""

    def test_detects_format_c(self, extract):
        result = extract.detect_format(FORMAT_C_README)
        assert result == "C"

    def test_parses_interventions_from_blurb(self, extract):
        interventions = extract.parse_format_c(FORMAT_C_README, FORMAT_C_SUBMISSION)
        # Should extract something from the blurb/prose
        assert len(interventions) >= 1

    def test_delta_bpb_is_null(self, extract):
        interventions = extract.parse_format_c(FORMAT_C_README, FORMAT_C_SUBMISSION)
        for i in interventions:
            assert i["delta_bpb"] is None

    def test_delta_source_is_submission_blurb(self, extract):
        interventions = extract.parse_format_c(FORMAT_C_README, FORMAT_C_SUBMISSION)
        for i in interventions:
            assert i["delta_source"] == "submission_blurb"


class TestFieldCoverage:
    """field_coverage = (non-null fields) / (submissions x 6)."""

    def test_perfect_coverage(self, extract):
        submissions = [
            {
                "submission_id": "test",
                "date": "2026-01-01",
                "author": "tester",
                "base_bpb": 1.13,
                "final_bpb": 1.12,
                "interventions": [{"name": "x", "category": "optimization", "delta_bpb": -0.01, "delta_source": "ablation_table"}],
                "parse_quality": "structured",
            }
        ]
        cov = extract.compute_field_coverage(submissions)
        assert cov == pytest.approx(1.0)

    def test_partial_coverage(self, extract):
        submissions = [
            {
                "submission_id": "test",
                "date": "2026-01-01",
                "author": "tester",
                "base_bpb": None,
                "final_bpb": 1.12,
                "interventions": [{"name": "x", "category": "optimization", "delta_bpb": None, "delta_source": "submission_blurb"}],
                "parse_quality": "minimal",
            }
        ]
        cov = extract.compute_field_coverage(submissions)
        # Missing: base_bpb (1 of 6 fields null) -> 5/6
        assert cov == pytest.approx(5.0 / 6.0, abs=0.01)

    def test_coverage_across_submissions(self, extract):
        submissions = [
            {
                "submission_id": "a",
                "date": "2026-01-01",
                "author": "x",
                "base_bpb": 1.0,
                "final_bpb": 1.0,
                "interventions": [{"name": "a", "category": "optimization", "delta_bpb": -0.01, "delta_source": "ablation_table"}],
                "parse_quality": "structured",
            },
            {
                "submission_id": "b",
                "date": None,
                "author": None,
                "base_bpb": None,
                "final_bpb": 1.0,
                "interventions": [],
                "parse_quality": "minimal",
            },
        ]
        cov = extract.compute_field_coverage(submissions)
        # a: 6/6, b: 2/6 (final_bpb + submission_id) -> 8/12
        assert cov == pytest.approx(8.0 / 12.0, abs=0.02)


class TestAppendExperiment:
    """--append-experiment mode with mock raw_runs.json."""

    def test_appends_experiment_entry(self, extract, tmp_path):
        raw_runs = {
            "experiment_id": "cycle_1_test",
            "intervention": "num_layers=12",
            "treatment_bpb": [1.120, 1.121, 1.119],
            "control_bpb": [1.125, 1.126, 1.124],
        }
        raw_path = tmp_path / "raw_runs.json"
        raw_path.write_text(json.dumps(raw_runs), encoding="utf-8")

        existing = {
            "submissions": [],
            "field_coverage": 0.0,
            "metadata": {"total_submissions": 0, "structured_count": 0, "prose_only_count": 0},
        }

        result = extract.append_experiment(existing, str(raw_path))
        assert len(result["submissions"]) == 1
        entry = result["submissions"][0]
        assert entry["submission_id"] == "cycle_1_test"
        assert entry["final_bpb"] == pytest.approx(1.12, abs=0.01)
        assert entry["base_bpb"] == pytest.approx(1.125, abs=0.01)
        assert len(entry["interventions"]) == 1
        assert entry["interventions"][0]["name"] == "num_layers=12"


class TestUnknownFormatFallback:
    """Unknown/empty README falls back gracefully."""

    def test_empty_readme(self, extract):
        result = extract.detect_format("")
        assert result == "C"

    def test_random_content(self, extract):
        result = extract.detect_format("Just some random text\nwith no structure\nat all.")
        assert result == "C"


class TestEndToEnd:
    """Integration: run extraction on the fixture records directory."""

    def test_extracts_all_three(self, extract, records_dir):
        result = extract.extract_all(str(records_dir))
        assert len(result["submissions"]) == 3

    def test_field_coverage_reasonable(self, extract, records_dir):
        result = extract.extract_all(str(records_dir))
        # With Format A and B providing structured data, coverage should be decent
        assert result["field_coverage"] >= 0.70

    def test_metadata_counts(self, extract, records_dir):
        result = extract.extract_all(str(records_dir))
        assert result["metadata"]["total_submissions"] == 3

    def test_output_json_roundtrip(self, extract, records_dir, tmp_path):
        result = extract.extract_all(str(records_dir))
        out = tmp_path / "interventions.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        reloaded = json.loads(out.read_text(encoding="utf-8"))
        assert reloaded["submissions"] == result["submissions"]
