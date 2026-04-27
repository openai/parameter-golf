"""Tests for submission assembly (T19)."""
import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.causal import submission_assembly as sa


# ========================== submission.json schema ==========================

class TestSubmissionJsonSchema:
    def test_required_fields_present(self, tmp_path):
        s = sa.build_submission_json(
            author="test_user",
            github_id="test_user",
            name="Test Record",
            blurb="A test submission",
            date="2026-03-25",
            val_loss=1.90,
            val_bpb=1.1250,
            bytes_total=15_000_000,
        )
        for field in ["author", "github_id", "name", "blurb", "date",
                       "val_loss", "val_bpb", "bytes_total"]:
            assert field in s, f"Missing field: {field}"

    def test_val_bpb_is_float(self):
        s = sa.build_submission_json(
            author="a", github_id="a", name="n", blurb="b",
            date="2026-01-01", val_loss=2.0, val_bpb=1.2, bytes_total=100,
        )
        assert isinstance(s["val_bpb"], float)


# ========================== artifact size check ============================

class TestArtifactSize:
    def test_under_limit_passes(self, tmp_path):
        f = tmp_path / "artifact.bin"
        f.write_bytes(b"\x00" * 15_000_000)
        assert sa.check_artifact_size(str(f), limit=16_000_000) is True

    def test_over_limit_fails(self, tmp_path):
        f = tmp_path / "artifact.bin"
        f.write_bytes(b"\x00" * 16_000_001)
        assert sa.check_artifact_size(str(f), limit=16_000_000) is False

    def test_exact_limit_passes(self, tmp_path):
        f = tmp_path / "artifact.bin"
        f.write_bytes(b"\x00" * 16_000_000)
        assert sa.check_artifact_size(str(f), limit=16_000_000) is True


# ========================== README required sections =======================

class TestReadmeValidation:
    GOOD_README = """## Record: Test (val_bpb: 1.1250)

### Results (3 seeds)

| Seed | val_bpb |
|------|---------|
| 42   | 1.1240  |
| 137  | 1.1250  |
| 256  | 1.1260  |

### Architecture

11 layers, 512-dim.

### Ablation Table

| Change | Base | This | Impact |
|--------|------|------|--------|
| Test   | 1.13 | 1.12 | -0.01  |
"""

    def test_valid_readme_passes(self):
        assert sa.validate_readme(self.GOOD_README) == []

    def test_missing_results_section(self):
        bad = "## Record\n### Architecture\nstuff\n### Ablation Table\nstuff"
        issues = sa.validate_readme(bad)
        assert any("Results" in i for i in issues)

    def test_missing_architecture_section(self):
        bad = "## Record\n### Results\nstuff\n### Ablation Table\nstuff"
        issues = sa.validate_readme(bad)
        assert any("Architecture" in i for i in issues)

    def test_missing_ablation_table(self):
        bad = "## Record\n### Results\nstuff\n### Architecture\nstuff"
        issues = sa.validate_readme(bad)
        assert any("Ablation" in i for i in issues)


# ========================== submission directory ==========================

class TestSubmissionDirectory:
    def test_all_required_files_present(self, tmp_path):
        # Create minimum submission directory
        out_dir = tmp_path / "2026-03-25_causal_1.1250"
        sa.assemble_submission(
            output_dir=str(out_dir),
            train_script_path="train_gpt.py",
            submission_json=sa.build_submission_json(
                author="test", github_id="test", name="Test", blurb="test",
                date="2026-03-25", val_loss=1.90, val_bpb=1.1250,
                bytes_total=15_000_000,
            ),
            readme_content=TestReadmeValidation.GOOD_README,
            seed_results=[
                {"seed": 42, "val_bpb": 1.1240},
                {"seed": 137, "val_bpb": 1.1250},
                {"seed": 256, "val_bpb": 1.1260},
            ],
        )
        assert (out_dir / "submission.json").exists()
        assert (out_dir / "README.md").exists()
        assert (out_dir / "train_gpt.py").exists()

    def test_submission_json_roundtrips(self, tmp_path):
        out_dir = tmp_path / "test_sub"
        sa.assemble_submission(
            output_dir=str(out_dir),
            train_script_path="train_gpt.py",
            submission_json=sa.build_submission_json(
                author="a", github_id="a", name="n", blurb="b",
                date="2026-01-01", val_loss=2.0, val_bpb=1.2,
                bytes_total=100,
            ),
            readme_content=TestReadmeValidation.GOOD_README,
            seed_results=[{"seed": 42, "val_bpb": 1.2}],
        )
        with open(out_dir / "submission.json") as f:
            loaded = json.load(f)
        assert loaded["val_bpb"] == 1.2


# ========================== seed results formatting ========================

class TestSeedResults:
    def test_mean_and_std(self):
        results = [
            {"seed": 42, "val_bpb": 1.1240},
            {"seed": 137, "val_bpb": 1.1250},
            {"seed": 256, "val_bpb": 1.1260},
        ]
        mean, std = sa.compute_seed_stats(results)
        assert abs(mean - 1.1250) < 0.0001
        assert abs(std - 0.001) < 0.001

    def test_single_seed_std_is_zero(self):
        results = [{"seed": 42, "val_bpb": 1.125}]
        mean, std = sa.compute_seed_stats(results)
        assert mean == 1.125
        assert std == 0.0


# ========================== causal findings README generation ==============

class TestCausalReadme:
    def test_generates_ablation_table(self):
        findings = [
            {"name": "Attention variant", "base_bpb": 1.13, "new_bpb": 1.12, "delta": -0.01},
        ]
        seeds = [
            {"seed": 42, "val_bpb": 1.1200},
            {"seed": 137, "val_bpb": 1.1210},
            {"seed": 256, "val_bpb": 1.1220},
        ]
        readme = sa.generate_readme(
            name="Causal Test",
            val_bpb=1.1210,
            findings=findings,
            seed_results=seeds,
            architecture_desc="11L 512-dim baseline",
        )
        assert "Ablation" in readme or "ablation" in readme
        assert "Results" in readme
        assert "Architecture" in readme
        assert "1.1200" in readme  # seed result
        assert "Attention variant" in readme

    def test_null_findings_uses_engineering_fallback(self):
        readme = sa.generate_readme(
            name="Engineering Fallback",
            val_bpb=1.1250,
            findings=[],  # null causal results
            seed_results=[{"seed": 42, "val_bpb": 1.1250}],
            architecture_desc="Baseline",
        )
        assert "engineering" in readme.lower() or "fallback" in readme.lower() or "baseline" in readme.lower()
